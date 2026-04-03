# -*- coding: utf-8 -*-
"""数据模型"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.database import Base


class ClassSession(Base):
    """一次课堂记录"""
    __tablename__ = "class_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), default="未命名课堂")
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    avg_attention = Column(Float, nullable=True)

    snapshots = relationship("AttentionSnapshot", back_populates="session", cascade="all, delete-orphan")


class AttentionSnapshot(Base):
    """每秒的专注度快照"""
    __tablename__ = "attention_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    elapsed_seconds = Column(Integer, default=0)
    total_people = Column(Integer, default=0)
    attention_score = Column(Float, default=0.0)
    behavior_counts = Column(JSON, default=dict)  # {"focused": 10, "head_down": 3, ...}

    session = relationship("ClassSession", back_populates="snapshots")
