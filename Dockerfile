# === Stage 1: Base Image ===
# Use an official NVIDIA CUDA base image with PyTorch compatibility
FROM pytorch/pytorch:2.0.1-cuda12.1-cudnn8-runtime

# === Environment Variables ===
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# === System Dependencies ===
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# === Python Configuration ===
# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip setuptools wheel

# === Application Setup ===
WORKDIR /app
# Copy the entire cloned project directory into the container
# Assumes you build from the parent of 'reproduction-project'
COPY reproduction-project/ /app/

# Install Python dependencies
WORKDIR /app
COPY reproduction-project/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory back to project root
WORKDIR /app

# === (Optional) Cache Hugging Face models ===
# Uncomment to pre-download heavy models and reduce startup latency
# ENV HF_HOME=/app/hf_cache
# RUN mkdir -p $HF_HOME && \
#     python -c "from transformers import AutoProcessor, AutoModelForVision2Seq;\
# import torch; MODEL='llava-hf/llava-interleave-qwen-7b-hf';\
# AutoProcessor.from_pretrained(MODEL, cache_dir='$HF_HOME', trust_remote_code=True);\
# AutoModelForVision2Seq.from_pretrained(MODEL, cache_dir='$HF_HOME', trust_remote_code=True)"

# === No fixed ENTRYPOINT ===
# We leave it flexible to run any script via `docker run <image> <script>.py`
CMD ["bash"] ["bash"]
