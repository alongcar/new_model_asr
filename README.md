# 多模型语音识别服务（Vosk / Whisper / Paraformer）

## 环境要求
- 操作系统：Windows 10/11（64位）
- Python：3.9（推荐）
- Pip：最新版本（`python -m pip install --upgrade pip`）
- CPU 或 GPU（可选）。若使用 GPU，请安装与你 CUDA 版本匹配的 `torch` 轮子。

## 本地模型准备
 - Vosk：将中文模型解压到 `model/vosk-model-small-cn-0.22`（或在 `config/settings.py` 设置 `vosk_model_path`）
- Whisper：如使用本地权重，设置 `settings.whisper_model_path`
- Paraformer（本地目录结构需包含下列文件）：
  - `model/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/`
    - `am.mvn`
    - `configuration.json`
    - `model.pt`（或对应权重文件）
    - `config.yaml`（若有）

## 安装步骤
1. 创建并激活虚拟环境
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
2. 安装依赖
   - `pip install -r requirements.txt`

> 已在 Windows + Python 3.9 环境下验证以下版本组合兼容：
> - `modelscope==1.11.1`
> - `datasets==2.14.5`
> - `pyarrow==14.0.2`
> - `numpy==1.26.4`
> - `funasr==1.2.9`

## 运行服务
- 启动：`python main.py`
- 访问 REST：
  - `GET http://localhost:8000/api/models` 查询可用模型
  - `POST http://localhost:8000/api/session` 创建会话（可选参数 `model_type`）
- WebSocket（实时识别）：
  - `ws://localhost:8000/ws/asr?model_type=paraformer`
  - 二进制帧发送原始 `int16` PCM 数据，采样率与 `settings.sample_rate` 一致（默认 16k）

## Docker 部署
1. **构建镜像**
   ```bash
   docker build -t asr-service:v1 .
   ```

2. **运行容器**
   > 注意：需要将本地模型目录挂载到容器内的 `/app/model` 路径
   ```bash
   docker run -d \
     -p 8000:8000 \
     -v d:/jhj/zhizhu/digital_man/asr_py10/model:/app/model \
     --name asr-service \
     asr-service:v1
   ```

## GPU 部署
- 前置条件：宿主机安装 NVIDIA 显卡驱动（>=525）与 NVIDIA Container Toolkit（`--gpus` 支持）。
- 依赖选择：
  - 将 `requirements.txt` 中的 `torch` 修改为 `torch==2.1.1+cu121`。
  - 构建时使用 CUDA 索引：`--extra-index-url https://download.pytorch.org/whl/cu121`。
- 构建镜像（GPU 版）：
  ```bash
  docker build -t asr-service:gpu .
  ```
- 运行容器（启用 GPU）：
  ```bash
  docker run -d --gpus all \
    -p 8000:8000 \
    -v d:/jhj/zhizhu/digital_man/asr_py10/model:/app/model \
    --name asr-service-gpu \
    asr-service:gpu
  ```
- GPU 验证：
  ```bash
  docker exec -it asr-service-gpu python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
  ```
- 可选基础镜像（更完整的 CUDA 运行环境）：
  ```Dockerfile
  FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
  RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libsndfile1 ffmpeg git build-essential \
      && rm -rf /var/lib/apt/lists/*
  COPY requirements.txt .
  RUN pip3 install --no-cache-dir --default-timeout=1000 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
  COPY . /app
  WORKDIR /app
  EXPOSE 8000
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

## 常见问题
### Docker 构建网络超时
如果遇到 `failed to resolve source metadata` 或 `content size of zero` 错误，通常是 Docker Hub 访问受限导致。
请在 Docker Desktop 设置中配置镜像加速：
1. 打开 Docker Desktop -> Settings -> Docker Engine
2. 添加 `registry-mirrors` 配置：
```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://huecker.io",
    "https://dockerhub.timeweb.cloud",
    "https://noohub.ru"
  ]
}
```
3. 点击 "Apply & restart"

## 配置说明
- 修改 `config/settings.py` 控制：
  - `default_model`（默认识别模型：`vosk`/`whisper`/`paraformer`）
  - `sample_rate`（音频采样率，默认 16000）
  - `vosk_model_path`、`whisper_model_path`、`paraformer_model_path`

## 快速自检
- 管线导入自检：
  - `python -c "from modelscope.pipelines import pipeline; from modelscope.utils.constant import Tasks; print('OK')"`
- Paraformer本地加载自检：
  - `python -c "from modelscope.pipelines import pipeline; from modelscope.utils.constant import Tasks; import os; m=r'd:\\jhj\\zhizhu\\digital_man\\asr_py10\\model\\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'; print(os.path.isdir(m)); p=pipeline(task=Tasks.auto_speech_recognition, model=m); print(type(p))"`

## 常见问题
- `ImportError: cannot import name 'LargeList' from 'datasets'`：固定 `modelscope==1.11.1` 与 `datasets==2.14.5`。
- `pyarrow / numpy` ABI 报错：将 `numpy` 固定到 `<2`（如 `1.26.4`）并配套 `pyarrow==14.0.2`。
- 前端无识别结果：确保以二进制帧发送 `int16` PCM，分片累计约 `>32KB` 触发一次识别；或降低触发阈值。
