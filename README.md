# ClassSense — 课堂专注度实时感知系统

> 面向高校课堂场景的 AI 教学辅助系统：实时感知 + 数据沉淀 + 智能助教一体化。

---

## 一、项目简介

ClassSense 基于计算机视觉与大语言模型，为教师提供"看得见"的课堂反馈：

- 通过摄像头持续捕捉教室画面，使用 **YOLOv8-Pose** 提取每位学生的人体关键点；
- 规则引擎将关键点映射为五类可解释行为（**专注 / 低头 / 趴桌 / 举手 / 扭头**）；
- 多帧稳定性跟踪 + 时间窗平滑得出班级实时 **专注度评分**；
- **WebSocket** 持续推送到教师仪表盘，MJPEG 实时预览 AI 的识别效果；
- 课后自动汇总为 **可视化报告**；**LLM 助教** 基于过程数据生成班级反馈与教学建议。

全程本地推理，画面不出机，兼顾实用性与隐私保护。

## 二、系统架构

```
 ┌──────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐
 │ 摄像头集 │ → │ YOLOv8-Pose     │ → │ 行为分析引擎     │ → │ 专注度评分   │
 │ (多路)   │   │ (姿态关键点)    │   │ (规则 + 稳定跟踪)│   │ (时间窗平滑) │
 └──────────┘   └─────────────────┘   └──────────────────┘   └──────┬───────┘
                                                                    │
 ┌──────────┐         ┌────────────┐          ┌─────────────┐       │
 │ 教师浏览器 │ ◀WS─── │ FastAPI    │ ◀─ ─ ─ ─ │ SQLite 持久化 │ ◀───┤
 │ (仪表盘) │         │ (REST/WS)  │          │ (课堂/快照) │       │
 └──────────┘         └─────┬──────┘          └─────────────┘       │
                            │                                       │
                            └─── LLM Agent（周期性分析 + 建议）─────┘
```

## 三、技术栈

| 层级 | 关键技术 | 版本/备注 |
|---|---|---|
| 视觉推理 | **YOLOv8-Pose**（m/s/n 可切换） | Ultralytics 8.2，COCO-Keypoint 17 点 |
| 深度学习 | **PyTorch** | CUDA 自动检测；CPU 下自动 FP32、GPU 下 FP16 |
| 视频 IO | **OpenCV** (`cv2.VideoCapture`) | Windows 优先 DSHOW 后端 + `CAP_PROP_BUFFERSIZE=1` |
| 行为判定 | 纯 Python 规则引擎 | 肩宽归一化阈值 + IoU 轨迹平滑 |
| Web 后端 | **FastAPI** + Uvicorn (asgi) | Python 3.10+ 异步协程 |
| 实时推送 | **WebSocket** + **MJPEG** (`multipart/x-mixed-replace`) | 数据与预览分离 |
| 并发模型 | AI 推理跑 `threading.Thread`；Web IO 跑 `asyncio` | 两者通过 `run_coroutine_threadsafe` 桥接 |
| 数据持久化 | **SQLite** + `aiosqlite` + `SQLAlchemy 2.x (async)` | 课堂元数据 + 3 秒粒度快照 |
| 可视化 | 原生 **HTML / CSS / JavaScript** + **Chart.js** + **Lucide icons** | 零前端构建依赖 |
| 智能助教 | **LLM 代理**（兼容 OpenAI 协议端点） | 可配置模型名与 base_url |
| 报告生成 | **matplotlib** 渲染时间线 | 输出嵌入式图片 |

## 四、核心功能

- **多摄像头并行** — 每路独立抓帧线程 + 独立行为跟踪器，支持宿舍/教室多角度布控
- **实时行为识别** — 专注 / 低头 / 趴桌 / 举手 / 扭头 五类，规则可解释、可调参
- **稳定跟踪** — IoU 贪心关联 + 窗口内多数投票，抑制偶发抖动
- **时间窗专注度** — 平滑窗按秒计算，对 FPS 波动鲁棒
- **AI 预览流** — 服务端 MJPEG 直推，浏览器原生解码，零 JS 轮询开销
- **设置页** — 一键扫描可用摄像头、切换调试画面、配置 LLM 端点
- **智能助教** — 周期性聚合课堂数据调用 LLM，给出班级状态与策略建议
- **课后报告** — 专注度曲线 + 行为分布 + LLM 总结，课后即查
- **边缘推理** — 模型与权重本地加载，画面与帧数据不出本机

## 五、项目结构

```
ClassSense/
├── app/
│   ├── ai/
│   │   ├── pose_detector.py      # YOLOv8-Pose 封装 + 自动设备选择 + FP16
│   │   ├── behavior_analyzer.py  # 规则引擎（低头/趴桌/举手/扭头）
│   │   ├── person_tracker.py     # IoU 贪心匹配 + 窗口投票平滑
│   │   └── attention_tracker.py  # 多路摄像头 + 专注度聚合 + 调试渲染
│   ├── agents/                   # LLM 智能助教
│   ├── llm/                      # LLM 协议客户端
│   ├── report/                   # 课后报告渲染
│   ├── routers/
│   │   ├── api.py                # REST 接口 + MJPEG 调试流
│   │   ├── websocket.py          # /ws/attention 实时推送
│   │   ├── agent.py              # 智能助教接口
│   │   ├── pages.py              # 页面路由
│   │   └── settings.py           # 设置读写
│   ├── config.py                 # 静态参数
│   ├── runtime_config.py         # 运行时配置读写
│   ├── database.py / models.py   # SQLAlchemy 模型 + 异步会话
│   └── main.py                   # FastAPI 装配 + 生命周期
├── static/                       # 前端静态资源
├── templates/                    # Jinja2 模板
├── yolov8m-pose.pt               # 主用模型（默认）
├── yolov8s-pose.pt               # 备用模型（低算力）
├── yolov8n-pose.pt               # 最小模型（极限场景）
├── app_config.json               # 运行时配置
├── llm_config.json               # LLM 端点配置
├── requirements.txt
└── run.py                        # 启动入口
```

## 六、快速开始

### 环境要求

- Python 3.10+
- 摄像头（USB 外接或笔记本内置）
- 可选：NVIDIA GPU + CUDA 版 PyTorch（可将推理帧率提升 5 倍以上）

### 安装与运行

```bash
git clone https://github.com/Mai-xiyu/ClassSense.git
cd ClassSense

pip install -r requirements.txt

python run.py
```

启动后浏览器访问 http://localhost:8000 即可。

### 关键参数（`app/config.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `POSE_MODEL` | `yolov8m-pose.pt` | 姿态模型，可改 s/n 换速度 |
| `POSE_IMG_SIZE` | `832` | 推理分辨率，CPU 建议 768-896，GPU 可上 1280 |
| `POSE_DEVICE` | `None`（自动） | 设为 `cuda` 强制 GPU；`cpu` 强制 CPU |
| `POSE_HALF` | `True` | FP16 推理，仅对 GPU 生效 |
| `DETECTION_FPS` | `20` | 检测上限，真实吞吐以推理速度为准 |
| `MIN_KEYPOINT_CONF` | `0.30` | 单点置信度阈值 |
| `ATTENTION_SMOOTH_WINDOW` | `5` | 专注度时间窗（秒） |

### LLM 配置（`llm_config.json`）

支持任何兼容 OpenAI 协议的端点。将 `base_url` / `api_key` / `model` 改为你自己的配置即可。未配置时其它功能不受影响。

## 七、隐私与安全

- 所有推理与存储发生在本机，无任何画面/数据外传（除非你主动配置 LLM 端点并请求）
- 数据库仅存聚合指标（专注度、行为计数），**不保存原始图像**
- MJPEG 调试流仅在本地回环开启，生产部署可自行加鉴权

## 八、许可证

本项目仅用于学习与交流，保留所有权利。

