# -*- coding: utf-8 -*-
"""LLM 设置 API"""

import os

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

from app.llm.client import load_config, save_config, llm_client
from app import runtime_config

router = APIRouter(prefix="/api/settings")


class LLMConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    agent_interval: Optional[int] = None


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***" if key else ""
    return key[:3] + "*" * (len(key) - 6) + key[-3:]


@router.get("/llm")
async def get_llm_config():
    """获取 LLM 配置（API Key 脱敏）"""
    cfg = load_config()
    return {
        "enabled": cfg.get("enabled", False),
        "base_url": cfg.get("base_url", ""),
        "api_key_masked": _mask_key(cfg.get("api_key", "")),
        "has_key": bool(cfg.get("api_key")),
        "model": cfg.get("model", ""),
        "max_tokens": cfg.get("max_tokens", 800),
        "temperature": cfg.get("temperature", 0.7),
        "agent_interval": cfg.get("agent_interval", 120),
    }


@router.post("/llm")
async def update_llm_config(body: LLMConfigUpdate):
    """更新 LLM 配置"""
    cfg = load_config()
    for field in ["enabled", "base_url", "api_key", "model",
                   "max_tokens", "temperature", "agent_interval"]:
        value = getattr(body, field, None)
        if value is not None:
            cfg[field] = value
    save_config(cfg)
    return {"ok": True}


@router.post("/llm/test")
async def test_llm_connection():
    """测试 LLM 连接"""
    cfg = load_config()
    if not cfg.get("api_key") or not cfg.get("base_url"):
        return {"success": False, "error": "请先填写 API 地址和密钥"}

    # 临时启用以测试
    old_enabled = cfg.get("enabled")
    cfg["enabled"] = True
    save_config(cfg)

    try:
        result = await llm_client.chat_complete([
            {"role": "user", "content": "请回复两个字：成功"}
        ], max_tokens=20)
        return {"success": True, "response": (result or "")[:200]}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        cfg["enabled"] = old_enabled
        save_config(cfg)


# 预设列表（给前端用）
PRESETS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "zhipu": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "qwen": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
    },
    "local_vllm": {
        "name": "本地 vLLM",
        "base_url": "http://localhost:8080/v1",
        "model": "default",
    },
}


@router.get("/llm/presets")
async def get_presets():
    """获取预设模型列表"""
    return PRESETS


# ==========================================================================
# 运行时应用配置（摄像头选择 / 调试预览开关 / 行为平滑）
# ==========================================================================


class AppConfigUpdate(BaseModel):
    cameras: Optional[List[int]] = None
    debug_preview_enabled: Optional[bool] = None
    behavior_stability_frames: Optional[int] = None
    tracker_iou_threshold: Optional[float] = None
    privacy_mode: Optional[bool] = None


@router.get("/app")
async def get_app_config():
    """获取运行时应用配置。"""
    cfg = runtime_config.load()
    return cfg


@router.post("/app")
async def update_app_config(body: AppConfigUpdate):
    """更新运行时应用配置，并在运行中立即热更新调试开关与(可能的)摄像头选择提示。"""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    cfg = runtime_config.save(updates)

    # 调试开关可以热切换到当前 tracker
    from app.main import tracker
    if tracker is not None and "debug_preview_enabled" in updates:
        tracker.set_debug_enabled(cfg["debug_preview_enabled"])

    # 摄像头变更需要重启课堂才生效，前端需要提示
    requires_restart = False
    if "cameras" in updates and tracker is not None:
        current = getattr(tracker, "camera_indices", [])
        if list(current) != list(cfg["cameras"]):
            requires_restart = True

    return {"ok": True, "config": cfg, "requires_restart": requires_restart}


@router.get("/cameras/scan")
async def scan_cameras():
    """探测可用的本地摄像头索引（0..limit-1）。"""
    import cv2

    cfg = runtime_config.load()
    limit = cfg["camera_scan_limit"]

    backend = getattr(cv2, "CAP_DSHOW", None)
    backends = []
    if os.name == "nt" and backend is not None:
        backends.append(backend)
    backends.append(None)
    if backend is not None and backend not in backends:
        backends.append(backend)

    available = []
    misses_after_found = 0

    for idx in range(limit):
        opened = False
        for current_backend in backends:
            cap = cv2.VideoCapture(idx) if current_backend is None else cv2.VideoCapture(idx, current_backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, _ = cap.read()
            cap.release()
            if ok:
                opened = True
                break

        if opened:
            available.append({"index": idx, "label": "摄像头 %d" % idx})
            misses_after_found = 0
        elif available:
            misses_after_found += 1
            if misses_after_found >= 2:
                break

    # 即便没扫到也要保证至少能选默认 0
    if not available:
        available.append({"index": 0, "label": "摄像头 0（未检测到，启动时再次尝试）"})

    return {
        "available": available,
        "selected": cfg["cameras"],
    }
