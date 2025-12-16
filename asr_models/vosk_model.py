import json
import logging
import threading
import uuid
from typing import Dict, List, Optional

from vosk import Model, KaldiRecognizer

from .base_model import BaseASRModel

logger = logging.getLogger(__name__)

class VoskModel(BaseASRModel):
    """Vosk语音识别模型实现"""
    
    def __init__(self, model_path: str, sample_rate: float, hotwords: Optional[List[str]] = None):
        super().__init__("vosk", sample_rate, hotwords)
        self.model_path = model_path
        
        # 会话管理
        self.sessions: Dict[str, Dict] = {}
        self.session_recognizers: Dict[str, KaldiRecognizer] = {}
        self.session_lock = threading.Lock()
        self.session_process_locks: Dict[str, threading.Lock] = {}
        self.audio_buffers: Dict[str, List[bytes]] = {}
    
    def load_model(self):
        """加载Vosk模型"""
        if self.is_loaded:
            return
            
        logger.info(f"加载Vosk模型: {self.model_path}")
        self.model = Model(self.model_path)
        self.is_loaded = True
        logger.info("Vosk模型加载完成")
    
    def _create_recognizer(self) -> KaldiRecognizer:
        """创建识别器实例"""
        return KaldiRecognizer(self.model, self.sample_rate)
    
    def process_audio(self, session_id: str, audio_data: bytes) -> Optional[str]:
        """处理音频数据 (同步阻塞)"""
        # 追加到音频缓冲 (用于后续整句重识别)
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].append(audio_data)
            process_lock = self.session_process_locks.get(session_id)
            recognizer = self.session_recognizers.get(session_id)
            session = self.sessions.get(session_id)

        if not process_lock or not recognizer or not session:
            return None

        # 阻塞执行识别
        with process_lock:
            try:
                if recognizer.AcceptWaveform(audio_data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        with self.session_lock:
                            session["results"].append(text)
                            session["last_result"] = text
                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text:
                        with self.session_lock:
                            session["partial"] = partial_text
            except Exception as e:
                logger.warning(f"Vosk识别处理失败: {e}")
        
        # 返回当前部分结果
        return self.get_partial_result(session_id)
    
    def create_session(self) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())
        recognizer = self._create_recognizer()
        
        with self.session_lock:
            self.session_recognizers[session_id] = recognizer
            self.session_process_locks[session_id] = threading.Lock()
            self.sessions[session_id] = {
                "id": session_id,
                "results": [],
                "partial": "",
                "last_result": "",
                "recognizer": recognizer,
            }
            self.audio_buffers[session_id] = []
        
        logger.info(f"创建Vosk会话: {session_id}")
        return session_id
    

    
    def get_partial_result(self, session_id: str) -> str:
        """获取部分结果"""
        with self.session_lock:
            session = self.sessions.get(session_id)
            return (session or {}).get("partial", "")
    
    def get_final_results(self, session_id: str) -> List[str]:
        """获取最终结果"""
        with self.session_lock:
            session = self.sessions.get(session_id)
            return list((session or {}).get("results", []))
    
    def close_session(self, session_id: str) -> Optional[str]:
        """关闭会话"""
        with self.session_lock:
            recognizer = self.session_recognizers.get(session_id)
            session = self.sessions.get(session_id)
            process_lock = self.session_process_locks.get(session_id)

        if process_lock:
            process_lock.acquire()

        final_text = None
        try:
            if recognizer and session:
                try:
                    final = json.loads(recognizer.FinalResult())
                    text = final.get("text", "").strip()
                    if text:
                        session["results"].append(text)
                        session["last_result"] = text
                        final_text = text
                    buffer = b"".join(self.audio_buffers.get(session_id, []))
                    if buffer:
                        try:
                            re_rec = self._create_recognizer()
                            if re_rec.AcceptWaveform(buffer):
                                improved_final = json.loads(re_rec.FinalResult())
                                improved_text = improved_final.get("text", "").strip()
                                if improved_text and improved_text != final_text:
                                    session["results"].append(improved_text)
                                    session["last_result"] = improved_text
                                    final_text = improved_text
                        except Exception as _:
                            pass
                except Exception as e:
                    logger.warning(f"获取Vosk最终结果失败: {e}")
        finally:
            with self.session_lock:
                if session_id in self.session_recognizers:
                    del self.session_recognizers[session_id]
                if session_id in self.session_process_locks:
                    if process_lock:
                        try:
                            process_lock.release()
                        except RuntimeError:
                            pass
                    del self.session_process_locks[session_id]
                if session_id in self.sessions:
                    del self.sessions[session_id]
                if session_id in self.audio_buffers:
                    del self.audio_buffers[session_id]
            
            logger.info(f"关闭Vosk会话: {session_id}")
        
        return final_text
    
    def cleanup(self):
        """清理资源"""
        with self.session_lock:
            self.sessions.clear()
            self.session_recognizers.clear()
            self.session_process_locks.clear()
            self.audio_buffers.clear()