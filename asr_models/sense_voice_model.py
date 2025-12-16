import logging
import threading
import uuid
import os
import tempfile
import time
import re
from typing import Dict, List, Optional

import numpy as np
from scipy.io import wavfile
try:
    from funasr import AutoModel
except ImportError:
    AutoModel = None

from .base_model import BaseASRModel

logger = logging.getLogger(__name__)

class SenseVoiceModel(BaseASRModel):
    def __init__(self, model_path: str, sample_rate: float, hotwords: Optional[List[str]] = None):
        super().__init__("sense_voice", sample_rate, hotwords)
        self.model_path = model_path
        self.sessions: Dict[str, Dict] = {}
        self.session_lock = threading.Lock()
        self.audio_buffers: Dict[str, List[bytes]] = {}
        self.model = None

    def load_model(self):
        if self.is_loaded:
            return
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(self.model_path)
        
        if AutoModel is None:
            raise ImportError("Please install funasr to use SenseVoiceModel")

        logger.info(f"加载SenseVoice模型: {self.model_path}")
        try:
            self.model = AutoModel(
                model=self.model_path,
                device="cpu",
                disable_update=True,
                log_level="ERROR"
            )
            self.is_loaded = True
            logger.info("SenseVoice模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        with self.session_lock:
            self.sessions[sid] = {
                "id": sid,
                "results": [],
                "partial": "",
                "last_result": "",
                "created_at": time.time(),
            }
            self.audio_buffers[sid] = []
        logger.info(f"创建SenseVoice会话: {sid}")
        return sid

    def process_audio(self, session_id: str, audio_data: bytes) -> Optional[str]:
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            self.audio_buffers[session_id].append(audio_data)
            
            # 累积一定量的音频再处理 (约2秒)
            buffer_size = sum(len(c) for c in self.audio_buffers[session_id])
        
        # 32k bytes ~ 1秒 (16k * 2 bytes)
        # 64k bytes ~ 2秒
        if buffer_size > 64000:
             return self._process_audio_buffer(session_id)
        
        return None

    def _process_audio_buffer(self, session_id: str) -> Optional[str]:
        with self.session_lock:
            buffer = self.audio_buffers.get(session_id, [])
            if not buffer:
                return None
            combined = b"".join(buffer)
            self.audio_buffers[session_id] = [] # 清空缓冲
        
        if len(combined) < 16000: # 忽略小于0.5秒的片段
            return None

        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, int(self.sample_rate), np.frombuffer(combined, dtype=np.int16))
            tmp_path = f.name
        
        text = ""
        try:
            # SenseVoice 推理
            res = self.model.generate(
                input=tmp_path,
                cache={},
                language="auto", 
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            # res 格式: [{'key': '...', 'text': '...'}]
            if res and isinstance(res, list):
                text = res[0].get("text", "")
            elif isinstance(res, dict):
                 text = res.get("text", "")
            
            # 清理文本中的情感标签 (如 <|HAPPY|>)
            text = re.sub(r'<\|.*?\|>', '', text).strip()

        except Exception as e:
            logger.error(f"SenseVoice推理错误: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        if text:
            with self.session_lock:
                session = self.sessions.get(session_id)
                if session:
                    session["results"].append(text)
                    session["last_result"] = text
            logger.info(f"SenseVoice识别: {text}")
            return text
        return None

    def get_partial_result(self, session_id: str) -> str:
        # SenseVoice 不支持实时流式部分结果
        return ""

    def get_final_results(self, session_id: str) -> List[str]:
        with self.session_lock:
            s = self.sessions.get(session_id)
            return list((s or {}).get("results", []))

    def close_session(self, session_id: str) -> Optional[str]:
        # 处理剩余缓冲
        final_text = self._process_audio_buffer(session_id)
        
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
        logger.info(f"关闭SenseVoice会话: {session_id}")
        return final_text

    def cleanup(self):
        self.sessions.clear()
        self.audio_buffers.clear()