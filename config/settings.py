import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # 模型路径配置
    vosk_model_path: str = "model/vosk-model-small-cn-0.22"
    whisper_model_size: str = "base"  # 改为模型名称：tiny, base, small, medium, large
    whisper_model_path: str = "model/whisper-base/small.pt"  # 新增：模型文件存放路径
    paraformer_model_path: str = "model/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    # 音频配置
    sample_rate: float = 16000.0
    max_workers: int = 4

    # 热词配置
    hotwords: List[str] = [
        "是", "否", "有", "没有", "不知道",
        "头痛", "头晕", "恶心", "呕吐", "心慌",
        "麻醉", "手术", "病史", "过敏", "药物",
        "丙泊酚", "七氟烷", "罗库溴铵", "舒芬太尼",
        "全麻", "局麻", "椎管内麻醉", "神经阻滞",
        "医生", "看病", "高血压", "大夫", "麻醉",
        "手术史", "麻醉史", "心脏"
    ]

    # 默认模型
    default_model: str = "vosk"  # vosk 或 whisper

    class Config:
        env_file = ".env"


settings = Settings()