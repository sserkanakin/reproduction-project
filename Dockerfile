# === Stage 1: Base Image ===
# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

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
# Upgrade pip and install PyTorch with CUDA support
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio

# === Application Setup ===
# Since the Dockerfile lives inside `reproduction-project/`, copy the entire context
WORKDIR /app
COPY . /app

# Install Python dependencies from your project
RUN pip install --no-cache-dir -r requirements.txt

# === (Optional) Cache Hugging Face models ===
# Uncomment to pre-download heavy models into /app/hf_cache
# ENV HF_HOME=/app/hf_cache
# RUN mkdir -p $HF_HOME && \
#     python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
# import torch; MODEL='llava-hf/llava-interleave-qwen-7b-hf'; \
# AutoProcessor.from_pretrained(MODEL, cache_dir='$HF_HOME', trust_remote_code=True); \
# AutoModelForVision2Seq.from_pretrained(MODEL, cache_dir='$HF_HOME', trust_remote_code=True)"

# === Flexible CMD ===
# Default to bash so you can run any script: `docker run <image> python your_script.py`
CMD ["bash"]
