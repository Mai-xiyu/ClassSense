# -*- coding: utf-8 -*-
"""LLM 设置 API"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.llm.client import load_config, save_config, llm_client

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
