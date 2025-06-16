# Dockerfile for LLaVA LoRA fine-tuning on NVIDIA L4 (24 GB)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
COPY . /workspace

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch + CUDA
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install Flash-Attn (latest available)
RUN pip install flash-attn

# Install BitsAndBytes for 4-bit quantization
RUN pip install bitsandbytes

# Install HuggingFace libraries
RUN pip install transformers accelerate datasets

# Install LLaVA from GitHub
RUN pip install git+https://github.com/haotian-liu/LLaVA.git@main

# Other utilities
RUN pip install openai tqdm

# Make workspace volume
VOLUME ["/workspace"]

# Default to bash
CMD ["bash"]
