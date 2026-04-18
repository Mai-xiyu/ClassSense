# -*- coding: utf-8 -*-
"""REST API路由"""

import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.models import ClassSession, AttentionSnapshot, TranscriptSegment

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


@router.get("/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: int, db: AsyncSession = Depends(get_db)):
    """获取指定课堂的完整语音转写。"""
    stmt = (
        select(TranscriptSegment)
        .where(TranscriptSegment.session_id == session_id)
        .order_by(TranscriptSegment.start_seconds)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "start_seconds": r.start_seconds,
            "end_seconds": r.end_seconds,
            "text": r.text,
            "speaker": r.speaker,
        }
        for r in rows
    ]


@router.get("/current")
async def get_current_status():
    """获取当前实时状态（供前端轮询备用）"""
    from app.main import tracker
    if tracker is None:
        return {"status": "not_running"}
    return tracker.get_current()


@router.get("/debug/frame")
async def get_debug_frame():
    """返回当前检测线程生成的调试画面。"""
    from app.main import tracker, _is_class_active
    from app import runtime_config

    if tracker is None or not _is_class_active:
        return Response(status_code=204)

    if not runtime_config.is_debug_enabled():
        return Response(status_code=204)

    frame = tracker.get_debug_frame()
    if not frame:
        return Response(status_code=204)

    current = tracker.get_current()
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Debug-Total-People": str(current.get("total_people", 0)),
        },
    )


@router.get("/debug/stream")
async def stream_debug_frames():
    """以 MJPEG (multipart/x-mixed-replace) 持续推送调试画面，
    由浏览器原生解码，帧率接近摄像头原生帧率。"""
    from app.main import tracker as _tracker_ref
    from app import runtime_config

    boundary = "frame"

    async def generator():
        loop = asyncio.get_running_loop()
        # 目标 ~30 FPS；摄像头给不到就自然降
        interval = 1.0 / 30.0
        while True:
            from app.main import tracker, _is_class_active

            if tracker is None or not _is_class_active or not runtime_config.is_debug_enabled():
                await asyncio.sleep(0.2)
                continue

            frame = await loop.run_in_executor(None, tracker.get_debug_frame)
            if frame:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame + b"\r\n"
                )
            await asyncio.sleep(interval)

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=" + boundary,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
