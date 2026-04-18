# -*- coding: utf-8 -*-
"""语音转写 Worker —— 本地 faster-whisper + sounddevice。

设计要点：
- 与 AttentionTracker 同构：独立 Thread，不阻塞 asyncio 主循环
- 麦克风按 16kHz/单声道采集，按 ~6 秒滑窗 + 0.8 秒静音切分送 whisper
- 结果通过 on_segment 回调（同步）抛回主线程，由 main.py 入库 + 喂 AgentManager
- 任何依赖（faster-whisper / sounddevice / numpy）缺失时优雅降级为"未启用"
- 隐私模式下完全不启动；启动后切到隐私模式由 main 负责调 stop()
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable, Optional


SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_SECONDS = 6.0          # 每段最长音频
SILENCE_TAIL_SECONDS = 0.8    # 末尾持续静音多长就切段
SILENCE_RMS = 0.012           # 简易静音阈（归一化幅值）


def list_microphones():
    """枚举可用输入设备。失败返回空列表（依赖未装）。"""
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        return []
    out = []
    try:
        devices = sd.query_devices()
        default_in = None
        try:
            default_in = sd.default.device[0]  # type: ignore[index]
        except Exception:
            default_in = None
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            out.append({
                "index": idx,
                "name": dev.get("name", f"device {idx}"),
                "default": idx == default_in,
                "samplerate": int(dev.get("default_samplerate", SAMPLE_RATE) or SAMPLE_RATE),
            })
    except Exception:
        return []
    return out


def asr_dependencies_status() -> dict:
    """前端用：判断是否能开 ASR。"""
    status = {"sounddevice": False, "faster_whisper": False, "numpy": False, "error": ""}
    try:
        import sounddevice  # noqa: F401
        status["sounddevice"] = True
    except Exception as e:
        status["error"] = f"sounddevice: {e}"
    try:
        import numpy  # noqa: F401
        status["numpy"] = True
    except Exception as e:
        status["error"] = (status["error"] + f"; numpy: {e}").strip("; ")
    try:
        import faster_whisper  # noqa: F401
        status["faster_whisper"] = True
    except Exception as e:
        status["error"] = (status["error"] + f"; faster-whisper: {e}").strip("; ")
    status["ready"] = status["sounddevice"] and status["faster_whisper"] and status["numpy"]
    return status


class ASRWorker:
    """单例：上课时 start()，下课时 stop()。"""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._on_segment: Optional[Callable[[dict], None]] = None
        self._mic_index: Optional[int] = None
        self._model_size = "small"
        self._language = "zh"
        self._start_wall: float = 0.0
        self._running = False
        self._error: str = ""

    # ---- 生命周期 ----

    def is_running(self) -> bool:
        return self._running

    def last_error(self) -> str:
        return self._error

    def start(self, mic_index: Optional[int], model_size: str, language: str,
              on_segment: Callable[[dict], None]) -> bool:
        if self._running:
            return True
        status = asr_dependencies_status()
        if not status["ready"]:
            self._error = status["error"] or "依赖未装"
            return False
        self._mic_index = mic_index if (mic_index is not None and mic_index >= 0) else None
        self._model_size = model_size or "small"
        self._language = language or "zh"
        self._on_segment = on_segment
        self._stop_evt.clear()
        self._error = ""
        self._start_wall = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="ASRWorker", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if not self._running:
            return
        self._stop_evt.set()
        self._running = False
        # 不 join，让线程自行退出，避免下课卡顿

    # ---- 核心循环 ----

    def _run(self):
        try:
            import numpy as np
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except Exception as e:
            self._error = f"导入失败: {e}"
            self._running = False
            return

        # 加载模型（首次会下载到 ~/.cache/huggingface）
        try:
            print(f"[ASR] 加载模型 {self._model_size} ...")
            model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
            print("[ASR] 模型就绪")
        except Exception as e:
            self._error = f"模型加载失败: {e}"
            print(f"[ASR] {self._error}")
            self._running = False
            return

        block_size = int(SAMPLE_RATE * 0.2)  # 200ms 一块
        max_window_samples = int(SAMPLE_RATE * WINDOW_SECONDS)
        silence_tail_samples = int(SAMPLE_RATE * SILENCE_TAIL_SECONDS)

        buf = np.zeros(0, dtype=np.float32)
        silence_run = 0  # 末尾静音样本数

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=block_size,
                device=self._mic_index,
            ) as stream:
                while not self._stop_evt.is_set():
                    try:
                        chunk, _ = stream.read(block_size)
                    except Exception as e:
                        self._error = f"采集出错: {e}"
                        time.sleep(0.5)
                        continue
                    if chunk is None or len(chunk) == 0:
                        continue
                    mono = chunk[:, 0] if chunk.ndim > 1 else chunk
                    buf = np.concatenate([buf, mono])

                    # 末尾静音检测
                    rms = float(np.sqrt(np.mean(mono * mono)))
                    if rms < SILENCE_RMS:
                        silence_run += len(mono)
                    else:
                        silence_run = 0

                    should_flush = (
                        len(buf) >= max_window_samples
                        or (silence_run >= silence_tail_samples and len(buf) >= int(SAMPLE_RATE * 1.5))
                    )
                    if should_flush:
                        seg_audio = buf
                        buf = np.zeros(0, dtype=np.float32)
                        silence_run = 0
                        # 整段都太安静就跳过
                        if float(np.sqrt(np.mean(seg_audio * seg_audio))) < SILENCE_RMS:
                            continue
                        self._transcribe_and_emit(model, seg_audio)
        except Exception as e:
            self._error = f"音频流失败: {e}\n{traceback.format_exc()}"
            print(f"[ASR] {self._error}")
        finally:
            self._running = False
            print("[ASR] 线程退出")

    def _transcribe_and_emit(self, model, audio_np):
        end_wall = time.time()
        end_seconds = end_wall - self._start_wall
        start_seconds = max(0.0, end_seconds - len(audio_np) / SAMPLE_RATE)
        try:
            segments, _info = model.transcribe(
                audio_np,
                language=self._language,
                vad_filter=True,
                beam_size=1,
                condition_on_previous_text=False,
            )
            text = "".join(s.text for s in segments).strip()
        except Exception as e:
            print(f"[ASR] 转写失败: {e}")
            return
        if not text:
            return
        if self._on_segment is None:
            return
        try:
            self._on_segment({
                "start_seconds": round(start_seconds, 2),
                "end_seconds": round(end_seconds, 2),
                "text": text,
            })
        except Exception as e:
            print(f"[ASR] 回调失败: {e}")


asr_worker = ASRWorker()
