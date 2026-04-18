# -*- coding: utf-8 -*-
"""页面路由"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR
from app import runtime_config
import os

router = APIRouter()

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def _asset_version() -> str:
    """根据静态资源修改时间生成版本号，避免浏览器拿到旧缓存。"""
    css_path = os.path.join(BASE_DIR, "static", "css", "dashboard.css")
    js_path = os.path.join(BASE_DIR, "static", "js", "dashboard.js")
    latest_mtime = max(os.path.getmtime(css_path), os.path.getmtime(js_path))
    return str(int(latest_mtime))


@router.get("/")
async def dashboard_page(request: Request):
    """教师仪表盘主页"""
    cfg = runtime_config.load()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "asset_version": _asset_version(),
            "debug_preview_enabled": cfg["debug_preview_enabled"],
            "privacy_mode": cfg.get("privacy_mode", False),
        },
    )


@router.get("/report/{session_id}")
async def report_page(request: Request, session_id: int):
    """课后报告页面"""
    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={"session_id": session_id, "asset_version": _asset_version()},
    )


@router.get("/settings")
async def settings_page(request: Request):
    """系统设置页面"""
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={"asset_version": _asset_version()},
    )
