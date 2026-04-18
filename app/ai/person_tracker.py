# -*- coding: utf-8 -*-
"""跨帧人体追踪 + 行为平滑，降低单帧误判对统计的影响。

算法：
1. 用 IoU 匹配当前帧检测框与已跟踪人员；
2. 为每个跟踪人员维护最近 N 帧的行为，取众数作为稳定行为；
3. 行为只在连续 N 帧都判定为新行为后才切换，减少抖动。
"""

import collections
import itertools


class _Track(object):
    __slots__ = ("id", "bbox", "history", "stable_behavior", "missed", "raw_behavior")

    def __init__(self, track_id, bbox, behavior, window):
        self.id = track_id
        self.bbox = bbox
        self.history = collections.deque([behavior], maxlen=window)
        self.stable_behavior = behavior
        self.raw_behavior = behavior
        self.missed = 0


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class PersonBehaviorTracker(object):
    """一个摄像头一个实例：为每个稳定跟踪到的人提供平滑后的行为标签。"""

    def __init__(self, window=4, iou_threshold=0.35, max_missed=4):
        self.window = max(1, int(window))
        self.iou_threshold = float(iou_threshold)
        self.max_missed = max_missed
        self._tracks = []
        self._id_seq = itertools.count(1)

    def _stable_from_history(self, history):
        counts = collections.Counter(history)
        top, top_count = counts.most_common(1)[0]
        # 仅当窗口内占多数（>= ceil(window/2)）时才切换
        if top_count * 2 >= len(history):
            return top
        return None

    def update(self, detections):
        """detections: list of (bbox, behavior). 返回 list of (bbox, stable_behavior, track_id)."""
        unmatched = list(range(len(detections)))
        results = [None] * len(detections)
        used_tracks = set()

        # 贪心按 IoU 降序匹配
        pairs = []
        for t_idx, track in enumerate(self._tracks):
            for d_idx in unmatched:
                bbox, _ = detections[d_idx]
                score = _iou(track.bbox, bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, t_idx, d_idx))
        pairs.sort(reverse=True)

        matched_dets = set()
        for score, t_idx, d_idx in pairs:
            if t_idx in used_tracks or d_idx in matched_dets:
                continue
            track = self._tracks[t_idx]
            bbox, behavior = detections[d_idx]
            track.bbox = bbox
            track.raw_behavior = behavior
            track.history.append(behavior)
            track.missed = 0

            stable = self._stable_from_history(track.history)
            if stable is not None:
                track.stable_behavior = stable

            results[d_idx] = (bbox, track.stable_behavior, track.id)
            used_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # 未匹配检测 → 新轨迹，初始直接使用原始行为
        for d_idx in range(len(detections)):
            if d_idx in matched_dets:
                continue
            bbox, behavior = detections[d_idx]
            new_track = _Track(next(self._id_seq), bbox, behavior, self.window)
            self._tracks.append(new_track)
            results[d_idx] = (bbox, behavior, new_track.id)

        # 未匹配轨迹 → missed++，超限淘汰
        survivors = []
        for t_idx, track in enumerate(self._tracks):
            if t_idx in used_tracks:
                survivors.append(track)
                continue
            track.missed += 1
            if track.missed <= self.max_missed:
                survivors.append(track)
        self._tracks = survivors

        return results
