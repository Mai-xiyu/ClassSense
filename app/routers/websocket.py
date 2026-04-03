# -*- coding: utf-8 -*-
"""WebSocket路由 —— 实时推送专注度数据到前端"""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# 已连接的WebSocket客户端
_clients: list[WebSocket] = []


async def broadcast(data: dict):
    """向所有已连接的客户端广播数据"""
    if not _clients:
        return
    message = json.dumps(data, ensure_ascii=False)
    disconnected = []
    for ws in _clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _clients.remove(ws)


@router.websocket("/ws/attention")
async def attention_websocket(websocket: WebSocket):
    """专注度数据WebSocket端点"""
    await websocket.accept()
    _clients.append(websocket)
    try:
        while True:
            # 保持连接，客户端可以发控制消息
            data = await websocket.receive_text()
            # 目前不需要处理客户端消息
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _clients:
            _clients.remove(websocket)
