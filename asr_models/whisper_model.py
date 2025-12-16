import logging
import threading
import uuid
import wave
import tempfile
import os
from typing import Dict, List, Optional
import time

import whisper
import numpy as np
from scipy.io import wavfile

from .base_model import BaseASRModel

logger = logging.getLogger(__name__)


class WhisperModel(BaseASRModel):
    def __init__(self, model_size: str, sample_rate: float, hotwords: Optional[List[str]] = None,
                 model_path: Optional[str] = None):
        super().__init__("whisper", sample_rate, hotwords)
        self.model_size = model_size
        self.model_path = model_path

        # 关键修复：初始化会话管理属性
        self.sessions: Dict[str, Dict] = {}
        self.session_lock = threading.Lock()
        self.audio_buffers: Dict[str, List[bytes]] = {}

        # 调试统计信息
        self.processed_chunks = 0
        self.total_audio_bytes = 0
        self.last_activity_time = time.time()

    def load_model(self):
        if self.is_loaded:
            return

        logger.info(f"加载Whisper模型: {self.model_size}")

        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"使用本地模型文件: {self.model_path}")
                file_size = os.path.getsize(self.model_path)
                logger.info(f"模型文件大小: {file_size} 字节")
                self.model = whisper.load_model(self.model_path)
            else:
                logger.info(f"下载模型: {self.model_size}")
                self.model = whisper.load_model(self.model_size)

            self.is_loaded = True
            logger.info("Whisper模型加载完成")

            # 测试模型是否正常工作
            logger.info("测试模型识别功能...")
            try:
                # 创建一个短暂的静音音频进行测试
                test_audio = np.zeros(1600, dtype=np.int16)  # 0.1秒静音
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    wavfile.write(temp_file.name, int(self.sample_rate), test_audio)
                    result = self.model.transcribe(temp_file.name, language='zh')
                    os.unlink(temp_file.name)
                logger.info(f"模型测试完成，识别结果: '{result.get('text', '')}'")
            except Exception as e:
                logger.warning(f"模型测试失败（可能正常）: {e}")

        except Exception as e:
            logger.error(f"Whisper模型加载失败: {e}")
            # 添加更详细的错误信息
            import traceback
            logger.error(f"详细错误堆栈: {traceback.format_exc()}")
            raise

    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """将字节数据转换为音频数组"""
        # 将字节数据转换为numpy数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        print(f"音频数据转换: {len(audio_data)} 字节 -> {len(audio_array)} 采样点")
        return audio_array.astype(np.float32) / 32768.0

    def _process_audio_buffer(self, session_id: str) -> Optional[str]:
        """处理音频缓冲区并返回识别结果"""
        print(f"开始处理音频缓冲区，会话: {session_id}")

        with self.session_lock:
            buffer = self.audio_buffers.get(session_id, [])
            if not buffer:
                print(f"会话 {session_id} 的音频缓冲区为空")
                return None

            # 合并所有音频数据
            combined_audio = b''.join(buffer)
            buffer_chunks = len(buffer)
            print(f"合并 {buffer_chunks} 个音频块，总大小: {len(combined_audio)} 字节")

            # 清空缓冲区
            self.audio_buffers[session_id] = []

        if len(combined_audio) < 16000:  # 小于1秒的音频不处理
            print(f"音频数据过短 ({len(combined_audio)} 字节)，跳过处理")
            return None

        try:
            # 创建临时文件处理音频
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                print(f"创建临时文件: {temp_file.name}")

                # 转换为numpy数组
                audio_array = self._bytes_to_audio_array(combined_audio)
                print(f"音频数组形状: {audio_array.shape}, 数据类型: {audio_array.dtype}")

                # 保存为WAV文件
                wavfile.write(temp_file.name, int(self.sample_rate),
                              (audio_array * 32768).astype(np.int16))
                print(f"WAV文件已保存: {temp_file.name}")

                # 使用Whisper进行识别
                print("开始语音识别...")
                start_time = time.time()

                result = self.model.transcribe(
                    temp_file.name,
                    language='zh',
                    fp16=False,  # 明确禁用FP16，使用FP32
                    initial_prompt=" ".join(self.hotwords) if self.hotwords else None
                )

                processing_time = time.time() - start_time
                print(f"语音识别完成，耗时: {processing_time:.2f}秒")

                text = result.get('text', '').strip()
                logger.info(f"识别结果: '{text}'")

                # 清理临时文件
                os.unlink(temp_file.name)
                print("临时文件已清理")

                # 更新统计信息
                self.processed_chunks += 1
                self.total_audio_bytes += len(combined_audio)
                self.last_activity_time = time.time()

                return text if text else None

        except Exception as e:
            logger.error(f"Whisper识别失败: {e}")
            import traceback
            logger.error(f"识别错误详情: {traceback.format_exc()}")
            return None

    def create_session(self) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())

        with self.session_lock:
            self.sessions[session_id] = {
                "id": session_id,
                "results": [],
                "partial": "",
                "last_result": "",
                "created_at": time.time(),
                "audio_chunks_received": 0
            }
            self.audio_buffers[session_id] = []

        logger.info(f"创建Whisper会话: {session_id}")
        print(f"当前活跃会话数: {len(self.sessions)}")
        return session_id

    def process_audio(self, session_id: str, audio_data: bytes) -> Optional[str]:
        """处理音频数据"""
        print(f"处理音频数据，会话: {session_id}, 数据大小: {len(audio_data)} 字节")

        # 简单验证音频数据（可选，用于深度调试）
        if len(audio_data) < 100:  # 如果数据包异常的小
            logger.warning(f"音频数据包过小，可能为静音或无效数据: {len(audio_data)} 字节")
            # 可以尝试打印前几个字节的十六进制值
            print(f"数据包头: {audio_data[:20].hex() if audio_data else 'None'}")

        with self.session_lock:
            if session_id not in self.sessions:
                logger.warning(f"会话不存在: {session_id}")
                return None

            # 更新会话统计
            self.sessions[session_id]["audio_chunks_received"] += 1
            session_stats = self.sessions[session_id]

            # 添加到音频缓冲区
            self.audio_buffers[session_id].append(audio_data)
            buffer_size = sum(len(chunk) for chunk in self.audio_buffers[session_id])
            buffer_chunks = len(self.audio_buffers[session_id])

            print(f"会话 {session_id} 缓冲区: {buffer_chunks} 个块, {buffer_size} 字节")

            # 如果缓冲区足够大，进行处理
            if buffer_size > 32000:  # 2秒的音频
                print(f"缓冲区达到处理阈值，开始处理...")
                text = self._process_audio_buffer(session_id)
                if text:
                    self.sessions[session_id]["partial"] = text
                    self.sessions[session_id]["last_result"] = text
                    logger.info(f"识别到文本: '{text}'")
                    return text
                else:
                    print("识别未返回文本")
            else:
                print(f"缓冲区大小 {buffer_size} 未达到阈值 32000")

        partial_result = self.get_partial_result(session_id)
        print(f"返回部分结果: '{partial_result}'")
        return partial_result

    def get_partial_result(self, session_id: str) -> str:
        """获取部分结果"""
        with self.session_lock:
            session = self.sessions.get(session_id)
            result = (session or {}).get("partial", "")
            print(f"获取会话 {session_id} 的部分结果: '{result}'")
            return result

    def get_final_results(self, session_id: str) -> List[str]:
        """获取最终结果"""
        with self.session_lock:
            session = self.sessions.get(session_id)
            results = list((session or {}).get("results", []))
            print(f"获取会话 {session_id} 的最终结果: {results}")
            return results

    def close_session(self, session_id: str) -> Optional[str]:
        """关闭会话"""
        logger.info(f"关闭会话: {session_id}")
        final_text = None

        with self.session_lock:
            if session_id in self.sessions:
                session_info = self.sessions[session_id]
                chunks_received = session_info.get("audio_chunks_received", 0)
                print(f"会话 {session_id} 共接收 {chunks_received} 个音频块")

                # 处理剩余的音频缓冲区
                if session_id in self.audio_buffers and self.audio_buffers[session_id]:
                    remaining_buffer_size = sum(len(chunk) for chunk in self.audio_buffers[session_id])
                    print(f"处理剩余音频缓冲区: {remaining_buffer_size} 字节")
                    final_text = self._process_audio_buffer(session_id)

                    if final_text:
                        self.sessions[session_id]["results"].append(final_text)
                        self.sessions[session_id]["last_result"] = final_text
                        logger.info(f"最终识别结果: '{final_text}'")

                # 清理会话
                if session_id in self.audio_buffers:
                    del self.audio_buffers[session_id]
                del self.sessions[session_id]

        logger.info(f"会话 {session_id} 已关闭")
        print(f"剩余活跃会话数: {len(self.sessions)}")
        return final_text

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        with self.session_lock:
            return {
                "model_loaded": self.is_loaded,
                "active_sessions": len(self.sessions),
                "processed_chunks": self.processed_chunks,
                "total_audio_bytes": self.total_audio_bytes,
                "last_activity": time.time() - self.last_activity_time,
                "session_details": {
                    session_id: {
                        "created": session_data.get("created_at", 0),
                        "chunks_received": session_data.get("audio_chunks_received", 0),
                        "results_count": len(session_data.get("results", [])),
                        "partial_result": session_data.get("partial", "")
                    }
                    for session_id, session_data in self.sessions.items()
                }
            }

    def cleanup(self):
        """清理资源"""
        logger.info("清理Whisper模型资源")
        # 记录清理前的统计信息
        debug_info = self.get_debug_info()
        logger.info(f"清理前统计: {debug_info}")

        # Whisper模型清理
        if hasattr(self, 'model'):
            del self.model
            logger.info("Whisper模型已清理")