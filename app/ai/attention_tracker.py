# -*- coding: utf-8 -*-
"""专注度追踪器 —— 汇总所有人的行为，计算班级整体专注度"""

import time
import threading
import collections
import cv2
import numpy as np

from app.ai.pose_detector import PoseDetector
from app.ai.behavior_analyzer import (
    analyze_behavior, ATTENTIVE_BEHAVIORS, BEHAVIOR_LABELS,
)
from app.config import CAMERA_INDEX, DETECTION_FPS, ATTENTION_SMOOTH_WINDOW


class AttentionTracker:
    """
    核心追踪器：摄像头 → 姿态检测 → 行为分类 → 专注度计算 → 推送数据
    在独立线程中运行，通过回调推送数据
    """

    def __init__(self):
        self.detector = PoseDetector()
        self.running = False
        self._thread = None
        self._callbacks = []

        # 专注度平滑缓冲
        self._score_buffer = collections.deque(
            maxlen=ATTENTION_SMOOTH_WINDOW * DETECTION_FPS
        )

        # 当前状态（线程安全读取）
        self._lock = threading.Lock()
        self._current_data = {
            "attention_score": 0.0,
            "smoothed_score": 0.0,
            "total_people": 0,
            "behavior_counts": {},
            "timestamp": 0,
        }

        # 运行时统计
        self.start_time = None
        self._all_snapshots = []

    def on_data(self, callback):
        """注册数据回调函数"""
        self._callbacks.append(callback)

    def get_current(self) -> dict:
        """获取当前最新数据"""
        with self._lock:
            return dict(self._current_data)

    def get_all_snapshots(self) -> list:
        """获取所有历史快照"""
        return list(self._all_snapshots)

    def start(self):
        """启动检测线程"""
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止检测"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        """检测主循环"""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[ERROR] 无法打开摄像头，请检查CAMERA_INDEX配置")
            self.running = False
            return

        interval = 1.0 / DETECTION_FPS

        try:
            while self.running:
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    continue

                # 1. 姿态检测
                persons = self.detector.detect(frame)

                # 2. 行为分类
                behavior_counts = {}
                for kpts in persons:
                    behavior = analyze_behavior(kpts)
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

                # 3. 计算专注度
                total = len(persons)
                if total > 0:
                    attentive = sum(
                        count for beh, count in behavior_counts.items()
                        if beh in ATTENTIVE_BEHAVIORS
                    )
                    raw_score = round(attentive / total * 100, 1)
                else:
                    raw_score = 0.0

                # 4. 平滑处理
                self._score_buffer.append(raw_score)
                smoothed = round(sum(self._score_buffer) / len(self._score_buffer), 1)

                # 5. 构造数据包
                elapsed = int(time.time() - self.start_time)
                data = {
                    "attention_score": raw_score,
                    "smoothed_score": smoothed,
                    "total_people": total,
                    "behavior_counts": behavior_counts,
                    "behavior_labels": {k: BEHAVIOR_LABELS.get(k, k) for k in behavior_counts},
                    "elapsed_seconds": elapsed,
                    "timestamp": time.time(),
                }

                with self._lock:
                    self._current_data = data

                self._all_snapshots.append(data)

                # 6. 调用回调推送
                for cb in self._callbacks:
                    try:
                        cb(data)
                    except Exception as e:
                        print(f"[WARN] 回调出错: {e}")

                # 控制帧率
                elapsed_loop = time.time() - loop_start
                sleep_time = interval - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cap.release()
