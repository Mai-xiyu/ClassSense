# -*- coding: utf-8 -*-
"""页面路由"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR
import os

router = APIRouter()

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@router.get("/")
async def dashboard_page(request: Request):
    """教师仪表盘主页"""
    return templates.TemplateResponse(request=request, name="dashboard.html")


@router.get("/report/{session_id}")
async def report_page(request: Request, session_id: int):
    """课后报告页面"""
    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={"session_id": session_id},
    )


@router.get("/settings")
async def settings_page(request: Request):
    """大模型设置页面"""
    return templates.TemplateResponse(request=request, name="settings.html")
