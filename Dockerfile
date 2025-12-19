# 使用官方 Python 3.10 轻量级镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# 替换 apt 源为阿里云（可选，加快国内构建速度）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装系统依赖
# libsndfile1:用于音频处理
# ffmpeg:用于音频格式转换
# build-essential:用于编译某些python库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    git \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 使用阿里云 pypi 源加速，增加超时时间
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cpu

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
# 假设模型挂载在 /app/model 目录下，通过环境变量覆盖 settings 中的默认路径
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]