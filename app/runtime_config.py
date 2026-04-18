# -*- coding: utf-8 -*-
"""运行时应用配置：摄像头选择 / 调试预览开关 / 行为平滑参数。

与 llm_config.json 分离，纯粹保存系统运行参数。
"""

import json
import os
import threading

from app.config import BASE_DIR


CONFIG_PATH = os.path.join(BASE_DIR, "app_config.json")

DEFAULTS = {
    # 摄像头：选中的索引列表，支持 1 个（单摄像头模式）或多个（多摄像头并用）
    "cameras": [0],
    # 调试预览面板是否在仪表盘显示
    "debug_preview_enabled": False,
    # 行为状态平滑帧数：同一人需连续 N 帧判定为新行为才更新，降低抖动误判
    "behavior_stability_frames": 4,
    # 跨帧同一人识别的 IoU 阈值
    "tracker_iou_threshold": 0.35,
    # 最多可扫描/使用的摄像头索引上限（用于扫描 0..N-1）
    "camera_scan_limit": 6,
}

_lock = threading.Lock()
_cache = None


def _sanitize(cfg: dict) -> dict:
    """清洗并补全配置项。"""
    merged = dict(DEFAULTS)
    if isinstance(cfg, dict):
        merged.update(cfg)

    # cameras 必须是去重的 int 列表，至少保留 [0]
    raw = merged.get("cameras", [0])
    if not isinstance(raw, list):
        raw = [raw]
    cleaned = []
    seen = set()
    for value in raw:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index < 0 or index in seen:
            continue
        seen.add(index)
        cleaned.append(index)
    if not cleaned:
        cleaned = [0]
    merged["cameras"] = cleaned

    merged["debug_preview_enabled"] = bool(merged.get("debug_preview_enabled"))

    try:
        frames = int(merged.get("behavior_stability_frames", 4))
    except (TypeError, ValueError):
        frames = 4
    merged["behavior_stability_frames"] = max(1, min(frames, 15))

    try:
        iou = float(merged.get("tracker_iou_threshold", 0.35))
    except (TypeError, ValueError):
        iou = 0.35
    merged["tracker_iou_threshold"] = max(0.1, min(iou, 0.9))

    try:
        scan = int(merged.get("camera_scan_limit", 6))
    except (TypeError, ValueError):
        scan = 6
    merged["camera_scan_limit"] = max(1, min(scan, 10))

    return merged


def load() -> dict:
    """读取当前运行时配置（带缓存）。"""
    global _cache
    with _lock:
        if _cache is not None:
            return dict(_cache)

        data = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception:
                data = {}

        _cache = _sanitize(data)
        return dict(_cache)


def save(updates: dict) -> dict:
    """增量更新并持久化运行时配置。"""
    global _cache
    with _lock:
        current = dict(_cache) if _cache is not None else {}
        if not current and os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
                    current = json.load(fp)
            except Exception:
                current = {}
        merged = _sanitize({**current, **(updates or {})})

        with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
            json.dump(merged, fp, indent=2, ensure_ascii=False)

        _cache = merged
        return dict(merged)


def get_cameras() -> list:
    return list(load()["cameras"])


def is_debug_enabled() -> bool:
    return bool(load()["debug_preview_enabled"])
