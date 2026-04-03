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
POSE_MODEL = "yolov8s-pose.pt"  # small模型，精度比nano高很多（mAP +10）
POSE_CONF_THRESHOLD = 0.35      # 置信度阈值（降低以检测到更多人）
POSE_IMG_SIZE = 960             # 推理分辨率（教室远景用大分辨率）
DETECTION_FPS = 3               # 每秒检测帧数（不需要太高）

# 行为判定阈值
HEAD_DOWN_ANGLE = 30            # 低头角度阈值（度）
LYING_RATIO_THRESHOLD = 0.3     # 趴桌判定：肩髋距离/站立时距离
HAND_RAISE_MARGIN = 50          # 举手判定：手腕高于肩膀的像素数
FACE_DIRECTION_THRESHOLD = 0.3  # 扭头判定：鼻子偏离中线比例

# 专注度平滑
ATTENTION_SMOOTH_WINDOW = 5     # 滑动平均窗口（秒）

# 服务器
HOST = "0.0.0.0"
PORT = 8000
