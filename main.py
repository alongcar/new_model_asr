#!/usr/bin/env python3
"""
多模型语音识别服务 - 支持Vosk和Whisper
"""

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse

from config.settings import settings
import importlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiModelASRService:
    """多模型语音识别服务"""
    
    def __init__(self):
        self.models: Dict[str, any] = {}
        self.sessions: Dict[str, Dict] = {}  # session_id -> {model_type, model_instance, session_data}
        self.session_lock = asyncio.Lock()
        # 统一的线程池，用于执行所有模型的阻塞推理任务
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
    
    async def initialize(self):
        """初始化所有模型"""
        logger.info("初始化语音识别模型...")
        VoskModel = None
        SenseVoiceModel = None
        ParaformerStreamingModel = None
        try:
            VoskModel = importlib.import_module("asr_models.vosk_model").VoskModel
        except Exception as e:
            logger.warning(f"Vosk模块导入失败: {e}")
        try:
            SenseVoiceModel = importlib.import_module("asr_models.sense_voice_model").SenseVoiceModel
        except Exception as e:
            logger.warning(f"SenseVoice模块导入失败: {e}")
        try:
            ParaformerStreamingModel = importlib.import_module("asr_models.paraformer_streaming_model").ParaformerStreamingModel
        except Exception as e:
            logger.warning(f"ParaformerStreaming模块导入失败: {e}")
        
        # 初始化Vosk模型
        try:
            if VoskModel is not None:
                self.models["vosk"] = VoskModel(
                    model_path=settings.vosk_model_path,
                    sample_rate=settings.sample_rate,
                    hotwords=settings.hotwords
                )
                self.models["vosk"].load_model()
                logger.info("Vosk模型初始化完成")
        except Exception as e:
            logger.error(f"Vosk模型初始化失败: {e}")
            self.models["vosk"] = None
        
        # 初始化Whisper模型
        # try:
        #     self.models["whisper"] = WhisperModel(
        #         model_size=settings.whisper_model_size,
        #         sample_rate=settings.sample_rate,
        #         hotwords=settings.hotwords,
        #         model_path=settings.whisper_model_path,
        #     )
        #     self.models["whisper"].load_model()
        #     logger.info("Whisper模型初始化完成")
        # except Exception as e:
        #     logger.error(f"Whisper模型初始化失败: {e}")
        #     self.models["whisper"] = None
        
        # 初始化Paraformer模型
        # try:
        #     self.models["paraformer"] = ParaformerModel(
        #         model_path=settings.paraformer_model_path,
        #         sample_rate=settings.sample_rate,
        #         hotwords=settings.hotwords
        #     )
        #     self.models["paraformer"].load_model()
        #     logger.info("Paraformer模型初始化完成")
        # except Exception as e:
        #     logger.warning(f"Paraformer模型初始化失败: {e}")

        # 初始化SenseVoice模型
        try:
            if SenseVoiceModel is not None:
                self.models["sense_voice"] = SenseVoiceModel(
                    model_path=settings.sense_voice_model_path,
                    sample_rate=settings.sample_rate,
                    hotwords=settings.hotwords
                )
                self.models["sense_voice"].load_model()
                logger.info("SenseVoice模型初始化完成")
        except Exception as e:
            logger.warning(f"SenseVoice模型初始化失败: {e}")

        # 初始化Paraformer Streaming模型
        try:
            if ParaformerStreamingModel is not None:
                self.models["paraformer_streaming"] = ParaformerStreamingModel(
                    model_path=settings.paraformer_streaming_model_path,
                    sample_rate=settings.sample_rate,
                    hotwords=settings.hotwords
                )
                self.models["paraformer_streaming"].load_model()
                logger.info("Paraformer Streaming模型初始化完成")
        except Exception as e:
            logger.warning(f"Paraformer Streaming模型初始化失败: {e}")
        #     self.models["paraformer"] = None
    
    async def create_session(self, model_type: str = None) -> str:
        """创建识别会话"""
        if model_type is None:
            model_type = settings.default_model
        
        if model_type not in self.models or self.models[model_type] is None:
            raise HTTPException(status_code=400, detail=f"模型 {model_type} 不可用")
        if not getattr(self.models[model_type], "is_loaded", False):
            raise HTTPException(status_code=400, detail=f"模型 {model_type} 未加载")
        
        session_id = str(uuid.uuid4())
        model_instance = self.models[model_type]
        
        # 在模型实例中创建会话
        model_session_id = model_instance.create_session()
        
        async with self.session_lock:
            self.sessions[session_id] = {
                "model_type": model_type,
                "model_instance": model_instance,
                "model_session_id": model_session_id,
                "created_at": asyncio.get_event_loop().time()
            }
        
        logger.info(f"创建会话: {session_id} (模型: {model_type})")
        return session_id
    
    async def process_audio(self, session_id: str, audio_data: bytes) -> Dict:
        """处理音频数据"""
        async with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="会话不存在")
        
        model_instance = session["model_instance"]
        model_session_id = session["model_session_id"]
        
        # 在统一线程池中执行阻塞的音频处理
        # 这样可以将所有模型的计算密集型任务从事件循环中移出
        loop = asyncio.get_running_loop()
        partial_result = await loop.run_in_executor(
            self.executor,
            model_instance.process_audio,
            model_session_id,
            audio_data
        )
        
        # 获取会话结果
        results = model_instance.get_final_results(model_session_id)
        
        return {
            "session_id": session_id,
            "model_type": session["model_type"],
            "partial_result": partial_result,
            "final_results": results,
            "last_result": results[-1] if results else ""
        }
    
    async def get_session_results(self, session_id: str) -> Dict:
        """获取会话结果"""
        async with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="会话不存在")
        
        model_instance = session["model_instance"]
        model_session_id = session["model_session_id"]
        
        partial_result = model_instance.get_partial_result(model_session_id)
        results = model_instance.get_final_results(model_session_id)
        
        return {
            "session_id": session_id,
            "model_type": session["model_type"],
            "partial_result": partial_result,
            "final_results": results,
            "last_result": results[-1] if results else ""
        }
    
    async def close_session(self, session_id: str) -> Dict:
        """关闭会话"""
        async with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="会话不存在")
            
            model_instance = session["model_instance"]
            model_session_id = session["model_session_id"]
            
            # 关闭模型会话
            final_result = model_instance.close_session(model_session_id)
            
            # 从会话管理中移除
            del self.sessions[session_id]
        
        logger.info(f"关闭会话: {session_id}")
        
        return {
            "session_id": session_id,
            "final_result": final_result,
            "message": "会话已关闭"
        }
    
    async def cleanup(self):
        """清理资源"""
        # 关闭所有会话
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            try:
                await self.close_session(session_id)
            except Exception as e:
                logger.error(f"关闭会话 {session_id} 失败: {e}")
        
        # 关闭统一线程池
        self.executor.shutdown(wait=True)
        
        # 清理模型资源
        for model_name, model_instance in self.models.items():
            if model_instance:
                try:
                    model_instance.cleanup()
                    logger.info(f"清理模型资源: {model_name}")
                except Exception as e:
                    logger.error(f"清理模型 {model_name} 失败: {e}")

# 全局服务实例
asr_service: Optional[MultiModelASRService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global asr_service
    
    # 启动服务
    asr_service = MultiModelASRService()
    await asr_service.initialize()
    
    yield
    
    # 关闭服务
    if asr_service:
        await asr_service.cleanup()

# 创建FastAPI应用
app = FastAPI(
    title="多模型语音识别服务",
    description="支持Vosk和Whisper的语音识别服务",
    lifespan=lifespan
)

@app.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket, model_type: str = "paraformer_streaming"):
    """WebSocket端点 - 实时音频流处理"""
    await websocket.accept()
    
    try:
        # 创建会话
        session_id = await asr_service.create_session(model_type)
        last_partial = ""
        last_results_len = 0
        last_final_text = ""
        await websocket.send_json({
            "type": "session_created",
            "session_id": session_id,
            "model_type": model_type
        })
        
        logger.info(f"WebSocket连接建立: {session_id} (模型: {model_type})")
        
        while True:
            # 接收音频数据
            data = await websocket.receive_bytes()
            print(f"DEBUG: WebSocket received {len(data)} bytes") # Keep this commented to avoid flood, user can uncomment if needed, or I should enable it? 
            # User asked "to see where the problem is", so maybe flooding is okay for a short debug session.
            # But high frequency logs might lag the terminal.
            # I'll enable it but maybe limit it or just enable it.
            if len(data) > 0:
                 pass # print(f"DEBUG: WebSocket received {len(data)} bytes") 
            
            if not data:
                continue
            
            # 处理音频
            result = await asr_service.process_audio(session_id, data)
            
            partial = result.get("partial_result")
            if partial:
                 print(f"DEBUG: Got partial from model: {partial}")

            if partial and partial != last_partial:
                print(f"DEBUG: Sending partial result: {partial}")
                await websocket.send_json({
                    "type": "partial_result",
                    "text": partial
                })
                last_partial = partial

            finals = result.get("final_results", [])
            if isinstance(finals, list) and len(finals) > last_results_len:
                for i in range(last_results_len, len(finals)):
                    text = finals[i]
                    if text and text != last_final_text:
                        print(f"DEBUG: Sending final result: {text}")
                        await websocket.send_json({
                            "type": "final_result",
                            "text": text
                        })
                        last_final_text = text
                last_results_len = len(finals)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if 'session_id' in locals():
            await asr_service.close_session(session_id)

@app.post("/api/session")
async def create_session(model_type: str = None):
    """创建新的识别会话"""
    session_id = await asr_service.create_session(model_type)
    return {"session_id": session_id, "model_type": model_type or settings.default_model}

@app.post("/api/recognize/{session_id}")
async def recognize_audio(session_id: str, audio_data: bytes):
    """识别音频数据"""
    result = await asr_service.process_audio(session_id, audio_data)
    return result

@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    """获取识别结果"""
    return await asr_service.get_session_results(session_id)

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """关闭会话"""
    return await asr_service.close_session(session_id)

@app.get("/api/models")
async def get_available_models():
    """获取可用模型列表"""
    available_models = []
    for model_name, model_instance in asr_service.models.items():
        if model_instance and model_instance.is_loaded:
            available_models.append(model_name)
    
    return {
        "available_models": available_models,
        "default_model": settings.default_model
    }

@app.get("/")
async def root():
    """根目录"""
    return {"message": "多模型语音识别服务", "supported_models": ["vosk", "whisper", "paraformer"]}

if __name__ == "__main__":
    # 启动服务
    logger.info("启动多模型语音识别服务...")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level
    )