# -*- coding: utf-8 -*-
"""Agent 相关 API"""

import asyncio
from fastapi import APIRouter
from sqlalchemy import select
from app.agents.manager import agent_manager
from app.llm.client import is_configured, has_credentials
from app.database import async_session
from app.models import AgentInsight
from app import runtime_config

router = APIRouter(prefix="/api/agent")


def _gate_error() -> str | None:
    """返回当前阻断 LLM 调用的原因；None 表示通过。"""
    if runtime_config.is_privacy_mode():
        return "隐私模式已开启，AI 助教已临时禁用。请先在设置中关闭隐私模式。"
    if not has_credentials():
        return "请先在设置中配置大模型 API"
    return None


@router.get("/status")
async def agent_status():
    """Agent 系统状态"""
    return {
        "configured": is_configured(),
        "privacy_mode": runtime_config.is_privacy_mode(),
        "has_credentials": has_credentials(),
        "has_student": agent_manager.latest["student"] is not None,
        "has_teacher": agent_manager.latest["teacher"] is not None,
        "has_summary": agent_manager.latest["summary"] is not None,
    }


@router.get("/latest")
async def agent_latest(session_id: int | None = None):
    """获取最新的 Agent 分析结果。传 session_id 时从 DB 读，避免跨课串。"""
    if session_id is None:
        return agent_manager.latest
    async with async_session() as db:
        result = {"student": None, "teacher": None, "summary": None}
        for agent in result.keys():
            stmt = (
                select(AgentInsight)
                .where(AgentInsight.session_id == session_id, AgentInsight.agent == agent)
                .order_by(AgentInsight.timestamp.desc())
                .limit(1)
            )
            row = (await db.execute(stmt)).scalar_one_or_none()
            if row is not None:
                result[agent] = row.content
        return result


@router.get("/history")
async def agent_history(session_id: int | None = None):
    """获取指定课堂的 Agent 分析历史。不传 session_id 则返回当前进行中课堂的内存记录。"""
    if session_id is None:
        return agent_manager.history
    async with async_session() as db:
        stmt = (
            select(AgentInsight)
            .where(AgentInsight.session_id == session_id)
            .order_by(AgentInsight.timestamp.asc())
        )
        rows = (await db.execute(stmt)).scalars().all()
        return [
            {
                "agent": r.agent,
                "content": r.content,
                "elapsed": r.elapsed_seconds,
                "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                "prompt_system": r.prompt_system,
                "prompt_user": r.prompt_user,
            }
            for r in rows
        ]


@router.post("/analyze")
async def trigger_analysis():
    """手动触发一次 Agent 分析（学生视角 + 教学顾问）"""
    err = _gate_error()
    if err:
        return {"error": err}
    try:
        await agent_manager.run_analysis()
        return {"ok": True, "latest": agent_manager.latest}
    except Exception as e:
        return {"error": str(e)}


@router.post("/summary")
async def trigger_summary():
    """
    课后汇总分析：异步后台启动 run_summary，
    结果通过 WebSocket (agent_chunk/agent_done/agent_error) 推送。
    """
    err = _gate_error()
    if err:
        return {"error": err}
    if not agent_manager._recent_data:
        return {"error": "没有可分析的课堂数据（请先上一节课）"}
    # 不 await，避免 HTTP 请求阻塞整个 LLM 流式耗时；WS 负责结果推送
    asyncio.create_task(agent_manager.run_summary())
    return {"ok": True}

