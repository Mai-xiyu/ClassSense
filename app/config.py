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
POSE_MODEL = "yolov8m-pose.pt"  # medium模型，mAP 67.0（比small +3，比nano +13）
POSE_CONF_THRESHOLD = 0.30      # 人体检测置信度（稍低，靠后续质量过滤兜底）
POSE_IOU_THRESHOLD = 0.50       # NMS IoU阈值（去除重叠框）
POSE_IMG_SIZE = 1280            # 推理分辨率（教室远景+多人，需要高分辨率）
DETECTION_FPS = 3               # 每秒检测帧数（medium模型3FPS足够）

# 关键点质量过滤
MIN_KEYPOINT_CONF = 0.40        # 单个关键点最低置信度
MIN_VISIBLE_KEYPOINTS = 5       # 每人至少可见关键点数（低于此丢弃，避免误判）

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
