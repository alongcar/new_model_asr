import logging
import threading
import uuid
import os
import time
from typing import Dict, List, Optional, Any

import numpy as np
try:
    from funasr import AutoModel
except ImportError:
    AutoModel = None

from .base_model import BaseASRModel

logger = logging.getLogger(__name__)

class ParaformerStreamingModel(BaseASRModel):
    def __init__(self, model_path: str, sample_rate: float, hotwords: Optional[List[str]] = None):
        super().__init__("paraformer_streaming", sample_rate, hotwords)
        self.model_path = model_path
        self.sessions: Dict[str, Dict] = {}
        self.session_lock = threading.Lock()
        self.model = None
        # 默认使用 paraformer-zh-streaming
        self.model_name_or_path = model_path if os.path.exists(model_path) else "paraformer-zh-streaming"

    def load_model(self):
        if self.is_loaded:
            return

        if AutoModel is None:
            raise ImportError("Please install funasr to use ParaformerStreamingModel")

        logger.info(f"加载Paraformer Streaming模型: {self.model_name_or_path}")
        try:
            # 加载流式模型
            # 这里的 chunk_size 是流式处理的关键配置 [0, 10, 5] 对应 [0, 600ms, 300ms]
            self.model = AutoModel(
                model=self.model_name_or_path,
                model_revision="v2.0.4",
                device="cpu", # 流式通常CPU够用，也可以设为 cuda
                disable_update=True,
                log_level="ERROR"
            )
            self.is_loaded = True
            logger.info("Paraformer Streaming模型加载完成")
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
                "cache": {}, # 初始化缓存
                "step": 0
            }
        logger.info(f"创建Paraformer Streaming会话: {sid}")
        return sid

    def _bytes_to_float32(self, audio_data: bytes) -> np.ndarray:
        arr = np.frombuffer(audio_data, dtype=np.int16)
        return arr.astype(np.float32) / 32768.0

    def process_audio(self, session_id: str, audio_data: bytes) -> Optional[str]:
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            session = self.sessions[session_id]
        
        # 转换音频格式
        audio_chunk = self._bytes_to_float32(audio_data)
        
        # 简单的VAD/静音检测（可选），这里直接送入模型
        # FunASR流式输入通常需要一定的长度，但AutoModel内部会处理
        
        try:
            # 执行流式推理
            # input: 音频数据
            # cache: 上下文缓存
            # is_final: 是否结束
            res = self.model.generate(
                input=audio_chunk,
                cache=session["cache"],
                is_final=False,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4, 
                decoder_chunk_look_back=1
            )
            
            # res 结构通常是List[Dict]，包含 'text' 等
            # [{'text': '...', 'mode': '2pass-online', ...}]
            
            if res and isinstance(res, list) and len(res) > 0:
                text = res[0].get("text", "").strip()
                if text:
                    # Paraformer Streaming 返回的是当前句子的累积结果
                    # 我们需要判断是更新部分结果还是产生了最终结果
                    # 注：2pass-online 模式下，它会不断修正
                    
                    with self.session_lock:
                        session["partial"] = text
                        session["last_result"] = text
                    
                    return text

        except Exception as e:
            pass

        return None

    def get_partial_result(self, session_id: str) -> str:
        with self.session_lock:
            s = self.sessions.get(session_id)
            return (s or {}).get("partial", "")

    def get_final_results(self, session_id: str) -> List[str]:
        with self.session_lock:
            s = self.sessions.get(session_id)
            return list((s or {}).get("results", []))

    def close_session(self, session_id: str) -> Optional[str]:
        final_text = ""
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            session = self.sessions[session_id]
        
        try:
            # 发送结束信号，触发 2pass 修正（如果有）
            res = self.model.generate(
                input=None,
                cache=session["cache"],
                is_final=True,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4, 
                decoder_chunk_look_back=1
            )
            
            if res and isinstance(res, list) and len(res) > 0:
                final_text = res[0].get("text", "").strip()
                if final_text:
                     with self.session_lock:
                        session["results"].append(final_text)
                        session["last_result"] = final_text
        except Exception as e:
            logger.error(f"关闭会话推理出错: {e}")

        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
        
        logger.info(f"关闭Paraformer Streaming会话: {session_id}, 最终结果: {final_text}")
        return final_text

    def cleanup(self):
        with self.session_lock:
            self.sessions.clear()