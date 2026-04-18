# -*- coding: utf-8 -*-
"""专注度追踪器 —— 单/多摄像头 → 姿态检测 → 行为平滑 → 专注度计算"""

import time
import threading
import collections
import os
import cv2
import numpy as np

from app.ai.pose_detector import PoseDetector
from app.ai.person_tracker import PersonBehaviorTracker
from app.ai.behavior_analyzer import (
    analyze_behavior, ATTENTIVE_BEHAVIORS, BEHAVIOR_LABELS,
)
from app.config import DETECTION_FPS, ATTENTION_SMOOTH_WINDOW, MIN_KEYPOINT_CONF
from app import runtime_config


DEBUG_BEHAVIOR_COLORS = {
    "focused": (74, 163, 22),
    "head_down": (4, 138, 202),
    "lying_down": (38, 38, 220),
    "hand_raised": (235, 99, 37),
    "looking_away": (237, 58, 124),
}
DEBUG_BEHAVIOR_TAGS = {
    "focused": "FOCUSED",
    "head_down": "HEAD_DOWN",
    "lying_down": "LYING",
    "hand_raised": "HAND",
    "looking_away": "AWAY",
}


class _CameraStream(object):
    """单个摄像头的后台抓帧线程，始终保留最新一帧。"""

    def __init__(self, index):
        self.index = index
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._opened = False
        self._running = False
        self._thread = None
        self._last_error = None
        self._read_failures = 0

    def _open_capture(self):
        backend = getattr(cv2, "CAP_DSHOW", None)
        preferred_backends = []
        if os.name == "nt" and backend is not None:
            preferred_backends.append(backend)
        preferred_backends.append(None)
        if backend is not None and backend not in preferred_backends:
            preferred_backends.append(backend)

        for current_backend in preferred_backends:
            cap = (
                cv2.VideoCapture(self.index)
                if current_backend is None
                else cv2.VideoCapture(self.index, current_backend)
            )
            if not cap.isOpened():
                cap.release()
                continue
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self._last_error = None
            return cap

        self._last_error = "无法打开摄像头 %d" % self.index
        return None

    def _reset_capture(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._opened = False

    def open(self):
        cap = self._open_capture()
        if cap is None:
            return False

        self._cap = cap
        self._opened = True
        self._running = True
        self._read_failures = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while self._running:
            if self._cap is None:
                self._cap = self._open_capture()
                if self._cap is None:
                    time.sleep(0.2)
                    continue
                self._opened = True

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._read_failures += 1
                if self._read_failures >= 10:
                    self._reset_capture()
                    self._read_failures = 0
                    time.sleep(0.1)
                    continue
                time.sleep(0.02)
                continue

            self._read_failures = 0
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._reset_capture()

    @property
    def opened(self):
        return self._opened

    @property
    def last_error(self):
        return self._last_error


class AttentionTracker(object):
    """核心追踪器：单/多摄像头并行 → 姿态检测 → 行为平滑 → 专注度计算。"""

    def __init__(self, camera_indices=None, debug_enabled=False):
        self.detector = PoseDetector()
        self.running = False
        self._thread = None
        self._callbacks = []
        self._debug_enabled = bool(debug_enabled)

        cfg = runtime_config.load()
        if camera_indices is None:
            camera_indices = cfg["cameras"]
        if not camera_indices:
            camera_indices = [0]
        self.camera_indices = list(camera_indices)

        self._stability_window = cfg["behavior_stability_frames"]
        self._iou_threshold = cfg["tracker_iou_threshold"]

        self._streams = [_CameraStream(idx) for idx in self.camera_indices]
        self._trackers = {
            idx: PersonBehaviorTracker(
                window=self._stability_window,
                iou_threshold=self._iou_threshold,
            )
            for idx in self.camera_indices
        }

        # 专注度平滑：用(时间戳, 分数)存储，按时间窗滚动，
        # 不再依赖 DETECTION_FPS——不管实际推理快慢，都保证 ATTENTION_SMOOTH_WINDOW 秒的窗口
        self._score_buffer = collections.deque()

        self._lock = threading.Lock()
        self._current_data = {
            "attention_score": 0.0,
            "smoothed_score": 0.0,
            "total_people": 0,
            "behavior_counts": {},
            "timestamp": 0,
            "cameras": list(self.camera_indices),
        }
        # 每个摄像头最近一次的检测结果，供预览按需渲染（解耦检测 FPS 与预览 FPS）
        self._stream_last_detection = {idx: ([], []) for idx in self.camera_indices}
        self._detection_lock = threading.Lock()

        self.start_time = None
        self._all_snapshots = []

    # ---------- public ----------

    def on_data(self, callback):
        self._callbacks.append(callback)

    def get_current(self):
        with self._lock:
            return dict(self._current_data)

    def get_all_snapshots(self):
        return list(self._all_snapshots)

    def get_debug_frame(self):
        """按需渲染调试画面，解耦检测 FPS 与预览 FPS。"""
        if not self._debug_enabled or not self.running:
            return None

        with self._detection_lock:
            snapshot = {
                idx: (list(people), list(behaviors))
                for idx, (people, behaviors) in self._stream_last_detection.items()
            }

        tiles = []
        for stream in self._streams:
            if not stream.opened:
                continue
            frame = stream.read()
            if frame is None:
                continue
            people, behaviors = snapshot.get(stream.index, ([], []))
            tile = self._render_camera_tile(frame, people, behaviors, "CAM %d" % stream.index)
            tiles.append(tile)

        if not tiles:
            return None

        canvas = self._compose_debug_grid(tiles)
        return self._encode_debug(canvas)

    def set_debug_enabled(self, enabled):
        self._debug_enabled = bool(enabled)

    def start(self):
        if self.running:
            return True
        opened_any = False
        for stream in self._streams:
            if stream.open():
                opened_any = True
            else:
                print("[WARN] %s" % stream.last_error)

        if not opened_any:
            print("[ERROR] 所有摄像头都无法打开，请检查设置中的摄像头选择")
            return False

        self.running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        for stream in self._streams:
            stream.close()

    # ---------- internal ----------

    def _render_camera_tile(self, frame, people, stable_behaviors, camera_label):
        preview = frame.copy()
        height, width = preview.shape[:2]

        overlay = preview.copy()
        cv2.rectangle(overlay, (0, 0), (width, 38), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.72, preview, 0.28, 0, preview)
        cv2.putText(
            preview,
            "%s  |  people: %d" % (camera_label, len(people)),
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for index, (person, behavior) in enumerate(zip(people, stable_behaviors), start=1):
            color = DEBUG_BEHAVIOR_COLORS.get(behavior, (255, 255, 255))
            bbox = person.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

                label = "#%d %s" % (index, DEBUG_BEHAVIOR_TAGS.get(behavior, "BODY"))
                label_width = max(70, 8 * len(label) + 10)
                label_top = max(0, y1 - 24)
                cv2.rectangle(preview, (x1, label_top), (x1 + label_width, label_top + 20), color, -1)
                cv2.putText(
                    preview,
                    label,
                    (x1 + 6, label_top + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            for x, y, conf in person["keypoints"]:
                if conf < MIN_KEYPOINT_CONF:
                    continue
                cv2.circle(preview, (int(x), int(y)), 3, color, -1)

        return preview

    def _compose_debug_grid(self, tiles):
        if not tiles:
            return None

        count = len(tiles)
        # 根据路数决定列数（1→1，2→2，3-4→2，5-6→3，7-9→3，10+→4）
        if count <= 1:
            cols = 1
        elif count <= 2:
            cols = 2
        elif count <= 6:
            cols = 2 if count <= 4 else 3
        elif count <= 9:
            cols = 3
        else:
            cols = 4
        rows = (count + cols - 1) // cols

        # 每格尺寸（保持 16:9）
        cell_w = 640
        cell_h = 360

        def letterbox(tile, dst_w, dst_h):
            h, w = tile.shape[:2]
            scale = min(dst_w / float(w), dst_h / float(h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((dst_h, dst_w, 3), dtype=resized.dtype)
            off_x = (dst_w - new_w) // 2
            off_y = (dst_h - new_h) // 2
            canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
            return canvas

        cells = [letterbox(t, cell_w, cell_h) for t in tiles]
        # 补齐空格
        while len(cells) < rows * cols:
            cells.append(np.zeros((cell_h, cell_w, 3), dtype=cells[0].dtype))

        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.hstack(cells[r * cols:(r + 1) * cols]))
        return np.vstack(row_imgs)

    def _encode_debug(self, canvas):
        if canvas is None:
            return None
        h, w = canvas.shape[:2]
        if w > 1280:
            scale = 1280.0 / w
            canvas = cv2.resize(canvas, (1280, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None
        return buf.tobytes()

    def _run_loop(self):
        interval = 1.0 / DETECTION_FPS

        # 等所有摄像头产生第一帧
        warmup_deadline = time.time() + 3.0
        while time.time() < warmup_deadline:
            if all(s.read() is not None for s in self._streams if s.opened):
                break
            time.sleep(0.05)

        try:
            while self.running:
                loop_start = time.time()

                total_people = 0
                aggregated_counts = {}
                per_stream_detection = {}

                for stream in self._streams:
                    if not stream.opened:
                        continue
                    frame = stream.read()
                    if frame is None:
                        continue

                    people = self.detector.detect_people(frame)

                    detections = []
                    for person in people:
                        behavior = analyze_behavior(person["keypoints"])
                        detections.append((person["bbox"], behavior))

                    tracker = self._trackers[stream.index]
                    smoothed = tracker.update(detections)

                    stable_behaviors = []
                    for i, result in enumerate(smoothed):
                        if result is None:
                            stable_behaviors.append(detections[i][1] if i < len(detections) else "focused")
                        else:
                            stable_behaviors.append(result[1])

                    total_people += len(people)
                    for beh in stable_behaviors:
                        aggregated_counts[beh] = aggregated_counts.get(beh, 0) + 1

                    per_stream_detection[stream.index] = (people, stable_behaviors)

                # 写入最新检测结果（预览按需读取）
                with self._detection_lock:
                    for idx in self.camera_indices:
                        if idx in per_stream_detection:
                            self._stream_last_detection[idx] = per_stream_detection[idx]

                if total_people > 0:
                    attentive = sum(
                        count for beh, count in aggregated_counts.items()
                        if beh in ATTENTIVE_BEHAVIORS
                    )
                    raw_score = round(attentive / total_people * 100, 1)
                else:
                    raw_score = 0.0

                now_ts = time.time()
                self._score_buffer.append((now_ts, raw_score))
                # 按时间窗滚动，丢弃超过 ATTENTION_SMOOTH_WINDOW 秒的历史
                cutoff = now_ts - ATTENTION_SMOOTH_WINDOW
                while self._score_buffer and self._score_buffer[0][0] < cutoff:
                    self._score_buffer.popleft()
                scores = [s for _, s in self._score_buffer]
                smoothed_score = round(sum(scores) / len(scores), 1) if scores else 0.0

                elapsed = int(time.time() - self.start_time)
                data = {
                    "attention_score": raw_score,
                    "smoothed_score": smoothed_score,
                    "total_people": total_people,
                    "behavior_counts": aggregated_counts,
                    "behavior_labels": {k: BEHAVIOR_LABELS.get(k, k) for k in aggregated_counts},
                    "elapsed_seconds": elapsed,
                    "timestamp": time.time(),
                    "cameras": list(self.camera_indices),
                }

                with self._lock:
                    self._current_data = data

                self._all_snapshots.append(data)

                for cb in self._callbacks:
                    try:
                        cb(data)
                    except Exception as e:
                        print("[WARN] 回调出错: %s" % e)

                elapsed_loop = time.time() - loop_start
                sleep_time = interval - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            for stream in self._streams:
                stream.close()
