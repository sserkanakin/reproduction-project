# === Stage 1: Base Image ===
# Use NVIDIA CUDA base image directly and install PyTorch via pip
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

# Upgrade pip and install PyTorch (with CUDA 12.1) + vision/audio libs
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio

# === Application Setup ===
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY eval-pipeline/ /app/eval-pipeline/
COPY generate_finetuning_data.py run_finetune.py eval-runner.py ./

# === No fixed ENTRYPOINT ===
# We leave it flexible to run any script via `docker run <image> <script>.py`
CMD ["bash"]
