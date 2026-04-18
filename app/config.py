# -*- coding: utf-8 -*-
"""全局配置"""

import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据库
DATABASE_URL = f"sqlite+aiosqlite:///{os.path.join(BASE_DIR, 'classroom.db')}"

# 摄像头
CAMERA_INDEX = 0  # 默认摄像头，0=内置，1=外接USB

# AI检测参数
POSE_MODEL = "yolov8m-pose.pt"  # medium模型（mAP 67），姿态关键点精度明显优于 s，对低头/趴桌/扭头判定至关重要
POSE_CONF_THRESHOLD = 0.40      # 人体检测置信度（略降避免漏检坐姿较侧的学生）
POSE_IOU_THRESHOLD = 0.50       # NMS IoU阈值（去除重叠框）
POSE_IMG_SIZE = 832             # 推理分辨率（m 模型 + 832 在 CPU 上约 3-4 FPS，比 1280 快 ~2.3 倍，精度几乎无损）
POSE_DEVICE = None              # 推理设备：None=自动（有CUDA就用GPU），'cuda'/'cpu'/'cuda:0' 可手动指定
POSE_HALF = True                # 启用FP16半精度推理（GPU上速度翻倍，CPU会自动回退FP32）
DETECTION_FPS = 20              # 每秒检测帧数上限（放宽让YOLO全速跑，实际吞吐由推理速度决定）

# 关键点质量过滤
MIN_KEYPOINT_CONF = 0.30        # 单个关键点最低置信度（降低阈值让鼻子/眼睛/耳朵更多参与行为判定）
MIN_VISIBLE_KEYPOINTS = 5       # 每人至少可见关键点数（远景学生关键点少，5 足够做行为判定）

# 行为判定阈值（均基于肩宽归一化）
HEAD_DOWN_ANGLE = 30            # 低头角度阈值（度）
LYING_RATIO_THRESHOLD = 0.3     # 趴桌判定：肩髋距离/站立时距离
HAND_RAISE_MARGIN = 50          # 举手判定：无肩宽时回退像素值
FACE_DIRECTION_THRESHOLD = 0.3  # 扭头判定：鼻子偏离中线比例

# 专注度平滑
ATTENTION_SMOOTH_WINDOW = 5     # 滑动平均窗口（秒）

# 服务器
HOST = "0.0.0.0"
PORT = 8000
