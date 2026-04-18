# -*- coding: utf-8 -*-
"""OpenAI 兼容的异步 LLM 客户端（支持 vLLM / DeepSeek / 智谱 / 通义千问等）"""

import json
import os

import httpx

from app.security import encrypt, decrypt, is_encrypted

# 配置文件路径
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "llm_config.json",
)

DEFAULT_CONFIG = {
    "enabled": False,
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "",
    "model": "deepseek-chat",
    "max_tokens": 800,
    "temperature": 0.7,
    "agent_interval": 120,
}


def _migrate_plaintext_if_needed(raw_cfg):
    """首次加载若发现 api_key 是明文，自动加密回写一次。"""
    key = raw_cfg.get("api_key") or ""
    if key and not is_encrypted(key):
        raw_cfg["api_key"] = encrypt(key)
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(raw_cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def load_config() -> dict:
    """读取 LLM 配置，api_key 解密后返回。"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _migrate_plaintext_if_needed(raw)
        cfg = dict(raw)
        if cfg.get("api_key"):
            cfg["api_key"] = decrypt(cfg["api_key"])
        return cfg
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    """保存 LLM 配置，api_key 加密后落盘。"""
    to_write = dict(config)
    if to_write.get("api_key"):
        to_write["api_key"] = encrypt(to_write["api_key"])
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(to_write, f, ensure_ascii=False, indent=2)


def is_configured() -> bool:
    """检查 LLM 是否已配置可用；隐私模式开启时强制视为未配置。"""
    # 隐私模式闸门：禁用所有 LLM 调用
    try:
        from app import runtime_config
        if runtime_config.load().get("privacy_mode"):
            return False
    except Exception:
        pass
    cfg = load_config()
    return bool(cfg.get("enabled") and cfg.get("api_key") and cfg.get("base_url"))


class LLMClient:
    """OpenAI Chat Completions 兼容客户端"""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        return self._client

    # ------ 非流式 ------
    async def chat_complete(self, messages: list[dict], **overrides) -> str | None:
        """
        发送非流式聊天请求，返回完整回复文本。
        如果未配置或出错则返回 None。
        """
        cfg = load_config()
        if not cfg.get("enabled") or not cfg.get("api_key"):
            return None

        client = self._ensure_client()
        url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        body = {
            "model": overrides.get("model", cfg.get("model", "deepseek-chat")),
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", cfg.get("max_tokens", 800)),
            "temperature": overrides.get("temperature", cfg.get("temperature", 0.7)),
            "stream": False,
        }

        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ------ 流式 ------
    async def chat_stream(self, messages: list[dict], **overrides):
        """
        发送流式聊天请求，yield 文本片段。

        Usage:
            async for chunk in llm_client.chat_stream(messages):
                print(chunk, end="")
        """
        cfg = load_config()
        if not cfg.get("enabled") or not cfg.get("api_key"):
            return

        client = self._ensure_client()
        url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        body = {
            "model": overrides.get("model", cfg.get("model", "deepseek-chat")),
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", cfg.get("max_tokens", 800)),
            "temperature": overrides.get("temperature", cfg.get("temperature", 0.7)),
            "stream": True,
        }

        async with client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# 全局单例
llm_client = LLMClient()
