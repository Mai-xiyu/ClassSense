# -*- coding: utf-8 -*-
"""Agent 相关 API"""

import asyncio
from fastapi import APIRouter
from app.agents.manager import agent_manager
from app.llm.client import is_configured

router = APIRouter(prefix="/api/agent")


@router.get("/status")
async def agent_status():
    """Agent 系统状态"""
    return {
        "configured": is_configured(),
        "has_student": agent_manager.latest["student"] is not None,
        "has_teacher": agent_manager.latest["teacher"] is not None,
        "has_summary": agent_manager.latest["summary"] is not None,
    }


@router.get("/latest")
async def agent_latest():
    """获取最新的 Agent 分析结果"""
    return agent_manager.latest


@router.get("/history")
async def agent_history():
    """获取本次课堂所有 Agent 分析历史"""
    return agent_manager.history


@router.post("/analyze")
async def trigger_analysis():
    """手动触发一次 Agent 分析（学生视角 + 教学顾问）"""
    if not is_configured():
        return {"error": "请先在设置中配置大模型 API"}
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
    if not is_configured():
        return {"error": "请先在设置中配置大模型 API"}
    if not agent_manager._recent_data:
        return {"error": "没有可分析的课堂数据（请先上一节课）"}
    # 不 await，避免 HTTP 请求阻塞整个 LLM 流式耗时；WS 负责结果推送
    asyncio.create_task(agent_manager.run_summary())
    return {"ok": True}

