# --- Base Image ---
# Example for Python 3.9 and CUDA 12.1 (adjust for your cloud target)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# For CPU-only cloud instances:
# FROM python:3.9-slim

# --- Environment Variables ---
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# --- System Dependencies & Python Installation ---
# For nvidia/cuda base images, Python often needs to be installed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.9 the default python and pip
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# --- Application Setup ---
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# Ensure requirements.txt specifies the correct PyTorch version for the CUDA version in the base image
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main application script from the project root to /app/ in the image
COPY eval-pipeline/eval-llava.py run_mmiu_llava_pipeline.py
# If you have other utility Python files in the root that this script imports, copy them too.
# e.g., COPY utils.py .

# --- (Optional) Download Hugging Face models during build ---
# This makes the image larger but ensures models are present.
# Useful if you want to avoid downloads on every container start, especially in restricted environments.
# Alternatively, mount a persistent Hugging Face cache volume at runtime.
# Example:
# ENV HF_HOME=/app/huggingface_cache
# RUN mkdir -p $HF_HOME
# RUN python -c "from transformers import AutoProcessor, LlavaForConditionalGeneration; \
#                MODEL_ID='llava-hf/llava-interleave-qwen-7b-hf'; \
#                AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir='$HF_HOME'); \
#                LlavaForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir='$HF_HOME')"

# --- Default command ---
# The script expects command-line arguments.
ENTRYPOINT ["python", "run_mmiu_llava_pipeline.py"]
# CMD ["--help"] # You can set default arguments here if desired (e.g., for testing)
