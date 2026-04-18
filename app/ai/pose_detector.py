# -*- coding: utf-8 -*-
"""YOLOv8-Pose 姿态检测器 —— 输入一帧画面，输出所有人的关键点"""

import numpy as np
from ultralytics import YOLO

from app.config import (
    POSE_MODEL, POSE_CONF_THRESHOLD, POSE_IOU_THRESHOLD,
    POSE_IMG_SIZE, MIN_KEYPOINT_CONF, MIN_VISIBLE_KEYPOINTS,
    POSE_DEVICE, POSE_HALF,
)


def _resolve_device():
    """自动选择推理设备：优先 CUDA，其次 MPS（Apple Silicon），最后 CPU。"""
    if POSE_DEVICE:
        return POSE_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


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
        self.device = _resolve_device()
        # 仅在 GPU 上启用 FP16（CPU 不支持）
        self.half = bool(POSE_HALF) and self.device.startswith("cuda")
        # 预热一次，避免首帧延迟高（编译/显存分配）
        try:
            import numpy as _np
            dummy = _np.zeros((POSE_IMG_SIZE, POSE_IMG_SIZE, 3), dtype=_np.uint8)
            self.model(dummy, imgsz=POSE_IMG_SIZE, device=self.device,
                       half=self.half, verbose=False)
            print("[INFO] PoseDetector 就绪 device=%s half=%s imgsz=%d"
                  % (self.device, self.half, POSE_IMG_SIZE))
        except Exception as e:
            print("[WARN] PoseDetector 预热失败（不影响使用）: %s" % e)

    def _build_bbox(self, person_kpts: np.ndarray, raw_bbox=None):
        """优先使用模型框；没有时退化为关键点包围盒。"""
        if raw_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in raw_bbox.tolist()]
            return (x1, y1, x2, y2)

        visible_points = person_kpts[person_kpts[:, 2] >= MIN_KEYPOINT_CONF][:, :2]
        if len(visible_points) == 0:
            return None

        min_xy = visible_points.min(axis=0)
        max_xy = visible_points.max(axis=0)
        padding = 12
        x1 = int(min_xy[0] - padding)
        y1 = int(min_xy[1] - padding)
        x2 = int(max_xy[0] + padding)
        y2 = int(max_xy[1] + padding)
        return (x1, y1, x2, y2)

    def detect_people(self, frame: np.ndarray):
        """检测并返回进入分析流程的人体元数据。"""
        results = self.model(
            frame,
            conf=POSE_CONF_THRESHOLD,
            iou=POSE_IOU_THRESHOLD,
            imgsz=POSE_IMG_SIZE,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        people = []
        for result in results:
            if result.keypoints is None:
                continue

            keypoints_batch = result.keypoints.data.cpu().numpy()  # shape: (N, 17, 3)
            boxes_batch = None
            if result.boxes is not None and result.boxes.xyxy is not None:
                boxes_batch = result.boxes.xyxy.cpu().numpy()

            for index, person_kpts in enumerate(keypoints_batch):
                visible = int(np.sum(person_kpts[:, 2] >= MIN_KEYPOINT_CONF))
                if visible < MIN_VISIBLE_KEYPOINTS:
                    continue

                # 结构化过滤：真人必须同时看到「面部线索」和「躯干线索」
                # 衣服上的印花/图案通常只满足其中一项，或关键点散乱不成人形
                face_kp_visible = any(
                    person_kpts[i, 2] >= MIN_KEYPOINT_CONF
                    for i in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
                )
                torso_kp_visible = any(
                    person_kpts[i, 2] >= MIN_KEYPOINT_CONF
                    for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
                )
                # 例外：双肩都清晰可见（趴桌典型特征——脸埋在手臂里但肩膀外露）
                # 此时即便 face_kp_visible=False 也要保留，让 behavior_analyzer 判定为 lying_down
                shoulders_both_clear = (
                    person_kpts[LEFT_SHOULDER, 2] >= MIN_KEYPOINT_CONF
                    and person_kpts[RIGHT_SHOULDER, 2] >= MIN_KEYPOINT_CONF
                )
                if not torso_kp_visible:
                    continue
                if not face_kp_visible and not shoulders_both_clear:
                    continue

                raw_bbox = None
                if boxes_batch is not None and index < len(boxes_batch):
                    raw_bbox = boxes_batch[index]

                # bbox 面积过小直接丢弃（印花/贴纸/小图案通常 < 60px 高）
                bbox = self._build_bbox(person_kpts, raw_bbox)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                if (x2 - x1) < 30 or (y2 - y1) < 60:
                    continue

                people.append({
                    "keypoints": person_kpts,
                    "bbox": bbox,
                    "visible_keypoints": visible,
                })

        return people

    def detect(self, frame: np.ndarray):
        """
        检测一帧画面中所有人的姿态关键点。

        Args:
            frame: BGR格式的OpenCV图像帧

        Returns:
            list[np.ndarray]: 每个人17个关键点的坐标，shape=(17, 3) -> (x, y, conf)
                              如果没检测到人则返回空列表
        """
        return [person["keypoints"] for person in self.detect_people(frame)]
