# -*- coding: utf-8 -*-
"""行为分析规则引擎 —— 根据关键点坐标判断每个人的行为状态"""

import math
import numpy as np

from app.ai.pose_detector import (
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
)
from app.config import (
    HEAD_DOWN_ANGLE, LYING_RATIO_THRESHOLD,
    HAND_RAISE_MARGIN, FACE_DIRECTION_THRESHOLD,
    MIN_KEYPOINT_CONF,
)


# 行为状态枚举
FOCUSED = "focused"         # 专注（抬头看前方）
HEAD_DOWN = "head_down"     # 低头
LYING_DOWN = "lying_down"   # 趴桌
HAND_RAISED = "hand_raised" # 举手
LOOKING_AWAY = "looking_away"  # 扭头/走神

BEHAVIOR_LABELS = {
    FOCUSED: "专注",
    HEAD_DOWN: "低头",
    LYING_DOWN: "趴桌",
    HAND_RAISED: "举手",
    LOOKING_AWAY: "扭头",
}

# 哪些行为算"专注"
ATTENTIVE_BEHAVIORS = {FOCUSED, HAND_RAISED}


def _valid(kpt, min_conf=None):
    """关键点是否有效（置信度足够）"""
    if min_conf is None:
        min_conf = MIN_KEYPOINT_CONF
    return kpt[2] > min_conf


def _distance(p1, p2):
    """两点之间的欧氏距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def analyze_behavior(keypoints: np.ndarray) -> str:
    """
    根据17个关键点坐标判断一个人的行为状态。
    判断优先级：趴桌 > 举手 > 低头 > 扭头 > 专注

    所有阈值使用肩宽归一化，适应不同距离/分辨率。
    课堂场景下髋部常被课桌遮挡，故趴桌/低头均有"无髋部"备选方案。

    Args:
        keypoints: shape=(17, 3) 每个关键点的 (x, y, confidence)

    Returns:
        行为状态字符串
    """
    nose = keypoints[NOSE]
    l_eye = keypoints[LEFT_EYE]
    r_eye = keypoints[RIGHT_EYE]
    l_shoulder = keypoints[LEFT_SHOULDER]
    r_shoulder = keypoints[RIGHT_SHOULDER]
    l_hip = keypoints[LEFT_HIP]
    r_hip = keypoints[RIGHT_HIP]
    l_wrist = keypoints[LEFT_WRIST]
    r_wrist = keypoints[RIGHT_WRIST]
    l_ear = keypoints[LEFT_EAR]
    r_ear = keypoints[RIGHT_EAR]

    # 预计算通用量
    has_shoulders = _valid(l_shoulder) and _valid(r_shoulder)
    has_nose = _valid(nose)
    l_eye_vis = _valid(l_eye)
    r_eye_vis = _valid(r_eye)
    eyes_hidden = not l_eye_vis and not r_eye_vis  # 双眼不可见 → 脸朝下

    # 上半身关键点质量门控：至少肩膀可见才做精确判定
    if not has_shoulders:
        return FOCUSED  # 信息不足，不瞎猜

    shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])

    # 肩宽过小说明人太远或检测质量差
    if shoulder_width < 15:
        return FOCUSED

    # ---- 1. 趴桌检测 ----
    # 方法A：有髋部——躯干纵向被严重压缩
    if _valid(l_hip) and _valid(r_hip):
        hip_y = (l_hip[1] + r_hip[1]) / 2
        torso_ratio = abs(hip_y - shoulder_mid_y) / shoulder_width
        if torso_ratio < 0.3:
            if has_nose and (nose[1] - shoulder_mid_y) / shoulder_width < 0.15:
                return LYING_DOWN
            if torso_ratio < 0.15:
                return LYING_DOWN

    # 方法B：无髋部（课桌遮挡）——通过鼻子位置 + 眼睛可见性推断
    if has_nose:
        nose_drop = (nose[1] - shoulder_mid_y) / shoulder_width
        # 鼻子明显低于肩膀中点 + 双眼不可见 → 趴在桌上
        if nose_drop > 0.25 and eyes_hidden:
            return LYING_DOWN
        # 鼻子几乎和肩膀齐平 + 双眼不可见 → 趴平
        if abs(nose_drop) < 0.1 and eyes_hidden:
            return LYING_DOWN

    # 方法C：整张脸都被遮挡（鼻子+双眼都不可见）而肩膀可见
    # → 极大概率是趴在手臂上（脸朝下埋在桌面）
    if not has_nose and eyes_hidden:
        return LYING_DOWN

    # ---- 2. 举手检测 ----
    # 使用肩宽归一化，远近都适用；肩宽无效时回退到固定像素
    raise_threshold = shoulder_width * 0.3 if shoulder_width > 15 else HAND_RAISE_MARGIN
    l_raised = (_valid(l_wrist) and _valid(l_shoulder)
                and l_wrist[1] < l_shoulder[1] - raise_threshold)
    r_raised = (_valid(r_wrist) and _valid(r_shoulder)
                and r_wrist[1] < r_shoulder[1] - raise_threshold)
    if l_raised or r_raised:
        return HAND_RAISED

    # ---- 3. 低头检测 ----
    if has_nose:
        head_drop = nose[1] - shoulder_mid_y
        relative_drop = head_drop / shoulder_width
        # 鼻子明显低于肩膀中点（看手机、做笔记）
        if relative_drop > 0.25:
            return HEAD_DOWN
        # 鼻子略低于肩膀 + 双眼不可见 → 面部朝下
        if relative_drop > 0.10 and eyes_hidden:
            return HEAD_DOWN
        # 只有一只眼可见也算低头（头已明显下垂侧歪）
        if relative_drop > 0.15 and (l_eye_vis != r_eye_vis):
            return HEAD_DOWN

    # ---- 4. 扭头检测 ----
    if has_nose:
        l_ear_visible = _valid(l_ear)
        r_ear_visible = _valid(r_ear)
        if l_ear_visible and r_ear_visible:
            ear_span = abs(l_ear[0] - r_ear[0])
            if ear_span > 10:
                mid_ear_x = (l_ear[0] + r_ear[0]) / 2
                nose_offset = abs(nose[0] - mid_ear_x) / ear_span
                # 原阈值 * 1.5 = 0.45 过严，降到 * 1.0 = 0.30 让日常扭头能触发
                if nose_offset > FACE_DIRECTION_THRESHOLD:
                    return LOOKING_AWAY
        elif l_ear_visible != r_ear_visible:
            # 只看到一只耳朵 → 侧转幅度较大
            if shoulder_width > 15:
                shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
                if abs(nose[0] - shoulder_mid_x) > shoulder_width * 0.3:
                    return LOOKING_AWAY
        # 两只耳朵都看不见 + 只有一只眼可见 → 明显扭头
        elif not l_ear_visible and not r_ear_visible and (l_eye_vis != r_eye_vis):
            return LOOKING_AWAY

    # ---- 5. 默认专注 ----
    return FOCUSED
