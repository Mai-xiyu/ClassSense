# -*- coding: utf-8 -*-
"""FastAPI 主应用"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR
from sqlalchemy import select, func, update

from app.database import init_db, async_session
from app.models import ClassSession, AttentionSnapshot
from app.ai.attention_tracker import AttentionTracker
from app.routers import websocket as ws_router
from app.routers import api as api_router
from app.routers import pages as page_router
from app.routers import agent as agent_router
from app.routers import settings as settings_router
from app.agents.manager import agent_manager
import os

# 全局追踪器实例
tracker: AttentionTracker = None
_current_session_id: int = None
_main_loop: asyncio.AbstractEventLoop = None
_is_class_active: bool = False  # 当前是否在上课


async def _push_data(data: dict):
    """将AI检测数据推送到WebSocket"""
    await ws_router.broadcast(data)


async def _save_snapshot(data: dict):
    """把快照数据存入数据库"""
    if _current_session_id is None:
        return
    async with async_session() as db:
        snapshot = AttentionSnapshot(
            session_id=_current_session_id,
            elapsed_seconds=data.get("elapsed_seconds", 0),
            total_people=data.get("total_people", 0),
            attention_score=data.get("smoothed_score", 0.0),
            behavior_counts=data.get("behavior_counts", {}),
        )
        db.add(snapshot)
        await db.commit()


def _on_detection_data_with_save(data: dict):
    """AI子线程的同步回调 → 跨线程调度到主事件循环"""
    if _main_loop is None or _main_loop.is_closed():
        return
    asyncio.run_coroutine_threadsafe(_push_data(data), _main_loop)
    if data.get("elapsed_seconds", 0) % 3 == 0:
        asyncio.run_coroutine_threadsafe(_save_snapshot(data), _main_loop)
    # 喂数据给 Agent 管理器
    agent_manager.feed_data(data)


async def start_class(name: str = None):
    """上课：创建课堂记录，启动AI检测"""
    global tracker, _current_session_id, _is_class_active

    if _is_class_active:
        return {"error": "课堂已在进行中"}

    # 创建课堂会话
    session_name = name or f"课堂 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    async with async_session() as db:
        session = ClassSession(name=session_name, start_time=datetime.now())
        db.add(session)
        await db.commit()
        await db.refresh(session)
        _current_session_id = session.id

    # 启动AI追踪器
    tracker = AttentionTracker()
    tracker.on_data(_on_detection_data_with_save)
    tracker.start()
    _is_class_active = True

    # 启动 Agent 定时分析
    agent_manager.start(_current_session_id, _main_loop)

    print(f"[上课] 课堂开始，ID: {_current_session_id}, 名称: {session_name}")
    return {"session_id": _current_session_id, "name": session_name}


async def stop_class():
    """下课：停止AI检测，保存数据，返回课堂ID供跳转报告"""
    global tracker, _current_session_id, _is_class_active

    if not _is_class_active:
        return {"error": "当前没有进行中的课堂"}

    # 停止追踪器和 Agent
    agent_manager.stop()
    tracker.stop()
    tracker = None
    _is_class_active = False

    session_id = _current_session_id

    # 计算并保存平均分
    async with async_session() as db:
        stmt = select(func.avg(AttentionSnapshot.attention_score)).where(
            AttentionSnapshot.session_id == session_id
        )
        result = await db.execute(stmt)
        avg = result.scalar()

        await db.execute(
            update(ClassSession)
            .where(ClassSession.id == session_id)
            .values(end_time=datetime.now(), avg_attention=avg)
        )
        await db.commit()

    _current_session_id = None
    print(f"[下课] 课堂结束，ID: {session_id}, 平均专注度: {avg}")

    # 推送下课消息到前端
    await ws_router.broadcast({
        "type": "class_ended",
        "session_id": session_id,
        "avg_attention": round(avg, 1) if avg else 0,
    })

    return {"session_id": session_id, "avg_attention": round(avg, 1) if avg else 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _main_loop

    _main_loop = asyncio.get_running_loop()
    await init_db()
    print("=" * 50)
    print('  AI课堂"读空气" — 教师教学效果实时感知系统')
    print("=" * 50)
    print("  仪表盘地址: http://localhost:8000")
    print('  打开浏览器，点击"上课"按钮开始检测')
    print("=" * 50)

    yield

    # 如果上课中直接关闭服务，也要保存
    if _is_class_active:
        await stop_class()
    print("[关闭] 系统已安全退出")


# 创建FastAPI应用
app = FastAPI(
    title="AI课堂读空气",
    description="教师教学效果实时感知系统",
    lifespan=lifespan,
)

# 挂载静态文件
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 注册路由
app.include_router(ws_router.router)
app.include_router(api_router.router)
app.include_router(page_router.router)
app.include_router(agent_router.router)
app.include_router(settings_router.router)
