import abc
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class BaseASRModel(abc.ABC):
    """ASR模型基础接口"""
    
    def __init__(self, model_name: str, sample_rate: float, hotwords: Optional[List[str]] = None):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.hotwords = hotwords or []
        self.is_loaded = False
    
    @abc.abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abc.abstractmethod
    def create_session(self) -> str:
        """创建识别会话"""
        pass
    
    @abc.abstractmethod
    def process_audio(self, session_id: str, audio_data: bytes) -> Optional[str]:
        """处理音频数据"""
        pass
    
    @abc.abstractmethod
    def get_partial_result(self, session_id: str) -> str:
        """获取部分识别结果"""
        pass
    
    @abc.abstractmethod
    def get_final_results(self, session_id: str) -> List[str]:
        """获取最终识别结果"""
        pass
    
    @abc.abstractmethod
    def close_session(self, session_id: str) -> Optional[str]:
        """关闭会话并返回最终结果"""
        pass
    
    @abc.abstractmethod
    def cleanup(self):
        """清理资源"""
        pass