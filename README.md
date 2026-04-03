# ClassSense — 课堂专注度实时感知系统

> **2026 年（第 19 届）中国大学生计算机设计大赛参赛作品**

---

## 项目简介

ClassSense 是一个基于计算机视觉的课堂教学辅助系统。通过摄像头实时捕捉学生姿态，利用 YOLOv8-Pose 提取人体关键点，经行为分析引擎判定学生状态（专注、低头、趴桌、举手、扭头），汇总为班级专注度指标，并通过 WebSocket 实时推送至教师仪表盘。

系统同时集成 LLM 智能助教模块，可对课堂数据进行深度分析并给出教学优化建议。

## 系统架构

```
摄像头 → YOLOv8-Pose 姿态估计 → 行为分析引擎 → 专注度评分
                                                      ↓
教师浏览器 ← WebSocket ← FastAPI 后端 ← SQLite 持久化
                                          ↓
                                    LLM Agent 分析
```

## 技术栈

| 层 | 技术 |
|---|---|
| AI 推理 | YOLOv8s-Pose（预训练，无需额外训练） |
| 后端 | Python 3.10+ / FastAPI / WebSocket / SQLAlchemy |
| 前端 | 原生 HTML + CSS + JavaScript / Chart.js |
| 数据库 | SQLite（aiosqlite 异步驱动） |
| 智能助教 | LLM API（可配置） |

## 核心功能

- **实时行为检测** — 3 FPS 持续分析课堂画面，输出学生行为分类与班级专注度
- **教师仪表盘** — 专注度环形指标 + 时序曲线 + 行为分布 + 事件日志
- **AI 助教洞察** — 基于 LLM 的班级反馈与教学优化建议
- **课后报告** — 自动生成专注度时间线与关键时段标注
- **隐私优先** — 不采集人脸、不上传原始画面，全部边缘计算

## 项目结构

```
ClassSense/
├── app/
│   ├── ai/                  # AI 核心模块
│   │   ├── pose_detector.py     # YOLOv8 姿态检测
│   │   ├── behavior_analyzer.py # 行为分析引擎
│   │   └── attention_tracker.py # 专注度追踪器
│   ├── agents/              # LLM Agent 模块
│   ├── llm/                 # LLM 客户端
│   ├── routers/             # FastAPI 路由
│   │   ├── api.py               # REST API
│   │   ├── websocket.py         # WebSocket 推送
│   │   ├── agent.py             # Agent 路由
│   │   ├── pages.py             # 页面路由
│   │   └── settings.py          # 设置路由
│   ├── report/              # 报告生成
│   ├── config.py            # 全局配置
│   ├── database.py          # 数据库
│   └── models.py            # 数据模型
├── static/
│   ├── css/dashboard.css    # 仪表盘样式
│   └── js/dashboard.js      # 仪表盘逻辑
├── templates/
│   ├── dashboard.html       # 主仪表盘
│   ├── report.html          # 课后报告
│   └── settings.html        # 系统设置
├── requirements.txt
├── run.py                   # 启动入口
└── llm_config.json          # LLM 配置
```

## 快速开始

### 环境要求

- Python 3.10+
- 摄像头（内置或 USB 外接）

### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/Mai-xiyu/ClassSense.git
cd ClassSense

# 安装依赖
pip install -r requirements.txt

# 启动服务（自动打开摄像头）
python run.py
```

启动后访问 http://localhost:8000 进入教师仪表盘。

### 配置说明

- **摄像头选择**: 修改 `app/config.py` 中的 `CAMERA_INDEX`（0=内置，1=外接）
- **检测参数**: 可在 `app/config.py` 调整推理分辨率、检测帧率、行为判定阈值等
- **LLM 配置**: 编辑 `llm_config.json` 配置智能助教的模型接口

## 许可证

本项目仅供学习与竞赛使用。
