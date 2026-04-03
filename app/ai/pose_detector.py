# -*- coding: utf-8 -*-
"""YOLOv8-Pose 姿态检测器 —— 输入一帧画面，输出所有人的关键点"""

import numpy as np
from ultralytics import YOLO

from app.config import POSE_MODEL, POSE_CONF_THRESHOLD, POSE_IMG_SIZE


# COCO关键点索引
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class PoseDetector:
    """基于YOLOv8-Pose的人体姿态检测器"""

    def __init__(self):
        self.model = YOLO(POSE_MODEL)

    def detect(self, frame: np.ndarray):
        """
        检测一帧画面中所有人的姿态关键点。

        Args:
            frame: BGR格式的OpenCV图像帧

        Returns:
            list[np.ndarray]: 每个人17个关键点的坐标，shape=(17, 3) -> (x, y, conf)
                              如果没检测到人则返回空列表
        """
        results = self.model(frame, conf=POSE_CONF_THRESHOLD, imgsz=POSE_IMG_SIZE, verbose=False)

        persons = []
        for result in results:
            if result.keypoints is None:
                continue
            kpts = result.keypoints.data.cpu().numpy()  # shape: (N, 17, 3)
            for person_kpts in kpts:
                persons.append(person_kpts)

        return persons
