# -*- coding: utf-8 -*-
"""Agent 管理器 —— 编排 Student / Teacher / Summary 三个 Agent 的分析周期"""

import asyncio
import collections
import time
from datetime import datetime

from app.llm.client import llm_client, load_config, is_configured
from app.agents.prompts import STUDENT_SYSTEM, TEACHER_SYSTEM, SUMMARY_SYSTEM
from app.routers import websocket as ws_router
from app.database import async_session
from app.models import AgentInsight

# 行为中文映射
_BEH_CN = {
    "focused": "专注",
    "head_down": "低头",
    "lying_down": "趴桌",
    "hand_raised": "举手",
    "looking_away": "扭头",
}


class AgentManager:
    """
    在课堂期间定时调用 LLM Agent 进行分析并推送结果。
    - 检测数据通过 feed_data() 从 AI 子线程同步喂入
    - 分析任务在主 asyncio 循环的后台协程中运行
    """

    def __init__(self):
        self._recent_data: collections.deque = collections.deque(maxlen=600)
        self._task: asyncio.Task | None = None
        self._running = False
        self._session_id: int | None = None
        # 最新洞察（供 API 读取）
        self.latest: dict = {"student": None, "teacher": None, "summary": None}
        # 所有历史洞察（供报告）
        self.history: list[dict] = []

    # ---------- 生命周期 ----------

    def start(self, session_id: int, loop: asyncio.AbstractEventLoop):
        """上课时启动"""
        self._session_id = session_id
        self._running = True
        self._recent_data.clear()
        self.latest = {"student": None, "teacher": None, "summary": None}
        self.history = []
        self._task = loop.create_task(self._periodic_loop())

    def stop(self):
        """下课时停止"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    def feed_data(self, data: dict):
        """从检测线程同步调用，喂入最新数据"""
        self._recent_data.append(data)

    # ---------- 定时循环 ----------

    async def _periodic_loop(self):
        """后台定时分析"""
        cfg = load_config()
        interval = max(cfg.get("agent_interval", 120), 30)
        # 等待第一波数据积累
        await asyncio.sleep(min(interval, 30))
        while self._running:
            try:
                if is_configured():
                    await self.run_analysis()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Agent] 分析出错: {e}")
            await asyncio.sleep(interval)

    # ---------- 核心分析 ----------

    async def run_analysis(self):
        """同时运行学生 Agent 和教师 Agent"""
        context = self._build_context()
        if context is None:
            return
        user_msg = self._format_user_message(context)

        # 并行运行两个 Agent
        await asyncio.gather(
            self._run_single("student", STUDENT_SYSTEM, user_msg),
            self._run_single("teacher", TEACHER_SYSTEM, user_msg),
        )

    async def run_summary(self, extra_stats: str = ""):
        """课后汇总分析"""
        if not is_configured():
            return None
        context = self._build_context()
        if context is None:
            return None
        raw_msg = self._format_user_message(context)

        parts = [raw_msg]
        if self.latest["student"]:
            parts.append(f"\n【学生视角Agent的分析】\n{self.latest['student']}")
        if self.latest["teacher"]:
            parts.append(f"\n【教学顾问Agent的分析】\n{self.latest['teacher']}")
        if extra_stats:
            parts.append(f"\n【课堂整体统计】\n{extra_stats}")

        user_msg = "\n".join(parts)

        try:
            await ws_router.broadcast({"type": "agent_start", "agent": "summary"})
            full_text = ""
            async for chunk in llm_client.chat_stream([
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ]):
                full_text += chunk
                await ws_router.broadcast({
                    "type": "agent_chunk",
                    "agent": "summary",
                    "content": chunk,
                })
            await ws_router.broadcast({
                "type": "agent_done",
                "agent": "summary",
                "content": full_text,
            })
            self.latest["summary"] = full_text
            self.history.append({
                "agent": "summary",
                "content": full_text,
                "timestamp": datetime.now().isoformat(),
                # 可解释性快照：保留发给 LLM 的 prompt，便于家长/督导事后复盘
                "prompt_system": SUMMARY_SYSTEM,
                "prompt_user": user_msg,
            })
            await self._persist("summary", full_text, SUMMARY_SYSTEM, user_msg, elapsed=0)
            return full_text
        except Exception as e:
            err = f"汇总分析失败: {e}"
            await ws_router.broadcast({"type": "agent_error", "agent": "summary", "error": err})
            return None

    async def _run_single(self, agent_name: str, system_prompt: str, user_msg: str):
        """运行单个 Agent（流式推送到前端）"""
        try:
            await ws_router.broadcast({"type": "agent_start", "agent": agent_name})
            full_text = ""
            async for chunk in llm_client.chat_stream([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]):
                full_text += chunk
                await ws_router.broadcast({
                    "type": "agent_chunk",
                    "agent": agent_name,
                    "content": chunk,
                })

            await ws_router.broadcast({
                "type": "agent_done",
                "agent": agent_name,
                "content": full_text,
            })
            elapsed = self._recent_data[-1].get("elapsed_seconds", 0) if self._recent_data else 0
            self.latest[agent_name] = full_text
            self.history.append({
                "agent": agent_name,
                "content": full_text,
                "elapsed": elapsed,
                "timestamp": datetime.now().isoformat(),
                # 可解释性快照：每条 AI 评论都带上系统/用户 prompt，便于质询时复盘
                "prompt_system": system_prompt,
                "prompt_user": user_msg,
            })
            await self._persist(agent_name, full_text, system_prompt, user_msg, elapsed)
        except Exception as e:
            err = f"{agent_name} 分析失败: {e}"
            await ws_router.broadcast({"type": "agent_error", "agent": agent_name, "error": err})

    # ---------- 持久化 ----------

    async def _persist(self, agent: str, content: str, sys_prompt: str, user_prompt: str, elapsed: int):
        """写入 DB。按 session_id 隔离，避免跨课串内容。"""
        if self._session_id is None:
            return
        try:
            async with async_session() as db:
                db.add(AgentInsight(
                    session_id=self._session_id,
                    agent=agent,
                    content=content,
                    elapsed_seconds=elapsed,
                    prompt_system=sys_prompt,
                    prompt_user=user_prompt,
                ))
                await db.commit()
        except Exception as e:
            print(f"[Agent] 持久化 insight 失败: {e}")

    # ---------- 上下文构建 ----------

    def _build_context(self) -> dict | None:
        if not self._recent_data:
            return None

        data_list = list(self._recent_data)
        latest = data_list[-1]
        total_points = len(data_list)

        avg_score = sum(d.get("smoothed_score", 0) for d in data_list) / total_points

        # 趋势判断
        half = total_points // 2
        if half > 3:
            recent_avg = sum(d.get("smoothed_score", 0) for d in data_list[half:]) / (total_points - half)
            earlier_avg = sum(d.get("smoothed_score", 0) for d in data_list[:half]) / half
            if recent_avg < earlier_avg - 10:
                trend = "下降"
            elif recent_avg > earlier_avg + 10:
                trend = "上升"
            else:
                trend = "平稳"
        else:
            trend = "数据积累中"

        # 最近事件
        events = []
        seen = set()
        for d in data_list[-30:]:
            beh = d.get("behavior_counts", {})
            total = d.get("total_people", 0)
            if total > 0:
                lying = beh.get("lying_down", 0)
                if lying > 0 and "趴桌" not in seen:
                    events.append(f"有{lying}人趴桌")
                    seen.add("趴桌")
                raised = beh.get("hand_raised", 0)
                if raised > 0 and "举手" not in seen:
                    events.append(f"有{raised}人举手")
                    seen.add("举手")
                hd = beh.get("head_down", 0)
                if hd > 0 and "低头" not in seen:
                    events.append(f"有{hd}人低头")
                    seen.add("低头")

        return {
            "elapsed_minutes": latest.get("elapsed_seconds", 0) // 60,
            "total_people": latest.get("total_people", 0),
            "attention_score": round(latest.get("smoothed_score", 0), 1),
            "behavior_counts": latest.get("behavior_counts", {}),
            "avg_score_period": round(avg_score, 1),
            "trend": trend,
            "recent_events": events,
        }

    @staticmethod
    def _format_user_message(ctx: dict) -> str:
        beh = ctx["behavior_counts"]
        beh_str = "、".join(
            f"{_BEH_CN.get(k, k)}{v}人" for k, v in beh.items() if v > 0
        ) or "暂无数据"

        events_str = "；".join(ctx["recent_events"]) if ctx["recent_events"] else "暂无特殊事件"

        return (
            f"当前课堂数据（已上课 {ctx['elapsed_minutes']} 分钟）：\n"
            f"- 总人数：{ctx['total_people']}\n"
            f"- 实时专注度：{ctx['attention_score']}%\n"
            f"- 本阶段平均专注度：{ctx['avg_score_period']}%\n"
            f"- 趋势：{ctx['trend']}\n"
            f"- 行为分布：{beh_str}\n"
            f"- 近期事件：{events_str}"
        )


# 全局单例
agent_manager = AgentManager()
