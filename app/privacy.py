# -*- coding: utf-8 -*-
"""隐私工具模块

提供差分隐私（Laplace 机制）与 k-匿名阈值两个基础工具，
用于发布班级/年级等聚合统计时抵御重识别攻击。

使用原则（已在 PRIVACY.md 锁定）：
1. 任何"分组样本量 < K_MIN"的统计不得发布具体数值。
2. 发布数值时应先按 Laplace(epsilon) 加噪。
3. epsilon 越小隐私越强但噪声越大；教育场景推荐 0.5~1.5。
"""

import random
import math

# --- 常量（如需调整应同步更新 PRIVACY.md）------------------------------------
K_MIN = 5              # 最小分组人数阈值（k-anonymity）
DEFAULT_EPSILON = 1.0  # Laplace 机制默认隐私预算


def k_anonymize(count, value, k_min=K_MIN, fallback="*"):
    """k-匿名阈值过滤。

    Args:
        count: 该分组的样本量
        value: 想要发布的数值（百分比、均值等）
        k_min: 最小阈值（默认 5）
        fallback: 不满足阈值时返回的占位（默认 '*'）

    Returns:
        value if count >= k_min else fallback
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        return fallback
    if count < k_min:
        return fallback
    return value


def laplace_noise(sensitivity=1.0, epsilon=DEFAULT_EPSILON):
    """生成一个服从 Laplace(0, sensitivity/epsilon) 的随机噪声。"""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    scale = float(sensitivity) / float(epsilon)
    # 反函数采样：L(0, b) = -b * sign(u) * ln(1 - 2|u|), u ~ U(-0.5, 0.5)
    u = random.random() - 0.5
    sign = 1 if u >= 0 else -1
    return -scale * sign * math.log(1 - 2 * abs(u) + 1e-12)


def dp_release(value, sensitivity=1.0, epsilon=DEFAULT_EPSILON, lo=None, hi=None):
    """对一个数值进行差分隐私发布：加噪后再裁剪到 [lo, hi]。

    Args:
        value: 原始聚合值
        sensitivity: 敏感度（一条记录改动能影响的最大范围，例如百分比场景 =100）
        epsilon: 隐私预算
        lo, hi: 可选裁剪范围（例如专注度 0~100）

    Returns:
        加噪并裁剪后的数值（float）
    """
    noisy = float(value) + laplace_noise(sensitivity, epsilon)
    if lo is not None:
        noisy = max(lo, noisy)
    if hi is not None:
        noisy = min(hi, noisy)
    return noisy


def safe_group_stat(count, value, sensitivity=100.0, epsilon=DEFAULT_EPSILON,
                    lo=0.0, hi=100.0, fallback="*"):
    """组合版：先检查 k-匿名再差分加噪。

    小于 K_MIN 直接返回 fallback，达标则返回加噪值。
    适合发布"男生/女生 / 某座位区 / 某学科"这类分组数据。
    """
    ok = k_anonymize(count, True, fallback=False)
    if not ok:
        return fallback
    return dp_release(value, sensitivity=sensitivity, epsilon=epsilon, lo=lo, hi=hi)
