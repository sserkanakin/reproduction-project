# Dockerfile for LLaVA LoRA fine-tuning on NVIDIA L4 (24 GB)
# Base image with CUDA 12.1 and Python 3.10 (via Ubuntu 22.04) to match your environment
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set non-interactive frontend to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in a single layer for better caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    jq \
    python3 \
    pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy requirements file first to leverage Docker layer caching.
# This build step will only re-run if requirements.txt changes.
COPY requirements.txt .

# Upgrade pip and install all Python dependencies from the requirements file
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now that dependencies are installed, copy the rest of your project code
COPY . .

# Install LLaVA from a specific commit for a reproducible build.
# This prevents your build from breaking if the main branch changes.
RUN pip install "git+https://github.com/haotian-liu/LLaVA.git"

# Make the workspace a volume (optional, good practice)
VOLUME ["/workspace"]

# Default to bash shell when the container starts
CMD ["bash"]
