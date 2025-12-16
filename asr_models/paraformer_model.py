import logging
import threading
import uuid
import os
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.io import wavfile
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from .base_model import BaseASRModel

logger = logging.getLogger(__name__)

class ParaformerModel(BaseASRModel):
    def __init__(self, model_path: str, sample_rate: float, hotwords: Optional[List[str]] = None):
        super().__init__("paraformer", sample_rate, hotwords)
        self.model_path = model_path
        self.sessions: Dict[str, Dict] = {}
        self.session_lock = threading.Lock()
        self.audio_buffers: Dict[str, List[bytes]] = {}
        self.processed_chunks = 0
        self.total_audio_bytes = 0
        self.last_activity_time = time.time()

    def load_model(self):
        if self.is_loaded:
            return
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(self.model_path)
        logger.info(f"加载Paraformer模型: {self.model_path}")
        try:
            # 设置10分钟超时（根据需要调整）

            self.pipeline = pipeline(
                task="auto-speech-recognition",
                model=self.model_path,
                device="cpu"  # 明确指定设备，后续可改为 "cuda"
            )
            self.is_loaded = True
            logger.info("Paraformer模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        # self.pipeline = pipeline(task=Tasks.auto_speech_recognition, model=self.model_path)
        # self.is_loaded = True
        logger.info("Paraformer模型加载完成")

    def _bytes_to_float32(self, audio_data: bytes) -> np.ndarray:
        arr = np.frombuffer(audio_data, dtype=np.int16)
        return arr.astype(np.float32) / 32768.0

    def _process_audio_buffer(self, session_id: str) -> Optional[str]:
        with self.session_lock:
            buffer = self.audio_buffers.get(session_id, [])
            if not buffer:
                return None
            combined = b"".join(buffer)
            self.audio_buffers[session_id] = []
        if len(combined) < int(self.sample_rate):
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio = self._bytes_to_float32(combined)
            wavfile.write(f.name, int(self.sample_rate), (audio * 32768).astype(np.int16))
            tmp_path = f.name
        try:
            res = self.pipeline(audio_in=tmp_path)
        finally:
            os.unlink(tmp_path)
        text = ""
        if isinstance(res, dict):
            text = str(res.get("text", "")).strip() or str(res.get("output", "")).strip()
        elif isinstance(res, str):
            text = res.strip()
        self.processed_chunks += 1
        self.total_audio_bytes += len(combined)
        self.last_activity_time = time.time()
        return text if text else None

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        with self.session_lock:
            self.sessions[sid] = {
                "id": sid,
                "results": [],
                "partial": "",
                "last_result": "",
                "created_at": time.time(),
                "audio_chunks_received": 0,
            }
            self.audio_buffers[sid] = []
        logger.info(f"创建Paraformer会话: {sid}")
        return sid

    def process_audio(self, session_id: str, audio_data: bytes) -> Optional[str]:
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            self.sessions[session_id]["audio_chunks_received"] += 1
            self.audio_buffers[session_id].append(audio_data)
            buffer_size = sum(len(c) for c in self.audio_buffers[session_id])
        if buffer_size > 32000:
            text = self._process_audio_buffer(session_id)
            if text:
                with self.session_lock:
                    self.sessions[session_id]["partial"] = text
                    self.sessions[session_id]["last_result"] = text
                logger.info(f"识别到文本: '{text}'")
                return text
        return self.get_partial_result(session_id)

    def get_partial_result(self, session_id: str) -> str:
        with self.session_lock:
            s = self.sessions.get(session_id)
            return (s or {}).get("partial", "")

    def get_final_results(self, session_id: str) -> List[str]:
        with self.session_lock:
            s = self.sessions.get(session_id)
            return list((s or {}).get("results", []))

    def close_session(self, session_id: str) -> Optional[str]:
        logger.info(f"关闭会话: {session_id}")
        final_text = None
        with self.session_lock:
            if session_id in self.sessions:
                if session_id in self.audio_buffers and self.audio_buffers[session_id]:
                    final_text = self._process_audio_buffer(session_id)
                    if final_text:
                        self.sessions[session_id]["results"].append(final_text)
                        self.sessions[session_id]["last_result"] = final_text
                self.audio_buffers.pop(session_id, None)
                self.sessions.pop(session_id, None)
        return final_text

    def cleanup(self):
        logger.info("清理Paraformer模型资源")
        if hasattr(self, "pipeline"):
            del self.pipeline