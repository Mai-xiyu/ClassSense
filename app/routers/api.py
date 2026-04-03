# -*- coding: utf-8 -*-
"""REST API路由"""

from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.models import ClassSession, AttentionSnapshot

router = APIRouter(prefix="/api")


class StartClassRequest(BaseModel):
    name: Optional[str] = None


@router.post("/start")
async def api_start_class(body: StartClassRequest = StartClassRequest()):
    """上课 — 开始AI检测"""
    from app.main import start_class
    result = await start_class(body.name)
    return result


@router.post("/stop")
async def api_stop_class():
    """下课 — 停止检测，保存数据"""
    from app.main import stop_class
    result = await stop_class()
    return result


@router.get("/status")
async def api_get_status():
    """获取当前系统状态"""
    from app.main import _is_class_active, _current_session_id
    return {
        "is_active": _is_class_active,
        "session_id": _current_session_id,
    }


@router.get("/sessions")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """获取所有课堂记录"""
    stmt = select(ClassSession).order_by(ClassSession.start_time.desc())
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "end_time": s.end_time.isoformat() if s.end_time else None,
            "avg_attention": s.avg_attention,
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}")
async def get_session_detail(session_id: int, db: AsyncSession = Depends(get_db)):
    """获取单次课堂的详细数据"""
    stmt = select(ClassSession).where(ClassSession.id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if not session:
        return {"error": "课堂记录不存在"}

    snap_stmt = (
        select(AttentionSnapshot)
        .where(AttentionSnapshot.session_id == session_id)
        .order_by(AttentionSnapshot.elapsed_seconds)
    )
    snap_result = await db.execute(snap_stmt)
    snapshots = snap_result.scalars().all()

    return {
        "id": session.id,
        "name": session.name,
        "start_time": session.start_time.isoformat() if session.start_time else None,
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "avg_attention": session.avg_attention,
        "snapshots": [
            {
                "elapsed_seconds": s.elapsed_seconds,
                "attention_score": s.attention_score,
                "total_people": s.total_people,
                "behavior_counts": s.behavior_counts,
            }
            for s in snapshots
        ],
    }


@router.get("/current")
async def get_current_status():
    """获取当前实时状态（供前端轮询备用）"""
    from app.main import tracker
    if tracker is None:
        return {"status": "not_running"}
    return tracker.get_current()
