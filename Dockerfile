# === Stage 1: Base Image ===
# Use an official NVIDIA CUDA base image.
# This example uses CUDA 12.1 and Ubuntu 22.04.
# Ensure this aligns with the PyTorch version you intend to install (e.g., PyTorch for cu121).
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# === Environment Variables ===
# Prevents Python from buffering stdout and stderr, making logs appear in real-time.
ENV PYTHONUNBUFFERED=1
# Prevents interactive prompts during apt-get installations.
ENV DEBIAN_FRONTEND=noninteractive
# Disables pip caching, which can be useful in Docker to reduce image size,
# but can also slow down builds if dependencies are frequently reinstalled.
# ENV PIP_NO_CACHE_DIR=off # Optional: uncomment if you prefer no cache
# Disables pip version check, minor optimization.
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# === System Dependencies ===
# Update package lists and install Python, pip, git, and other essentials.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    # Clean up apt cache to reduce image size
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# === Python Configuration ===
# Make python3.10 the default 'python3' and 'python'.
# Using a specific version like python3.10 is more explicit than relying on system default python3.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip for the selected Python version.
# Using 'python -m pip' ensures you're using the pip associated with the 'python' command (now python3.10).
RUN python -m pip install --upgrade pip setuptools wheel

# === Application Setup ===
# Set the working directory for subsequent commands.
WORKDIR /app

# Copy the requirements file. This is done before copying other app files
# to leverage Docker's layer caching if requirements don't change often.
COPY requirements.txt .

# Install Python dependencies specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the /app/ directory.
# This assumes your main script and any other necessary files are in the build context root.
COPY eval-pipeline/eval-llava.py eval-runner.py
# If you have a utils directory or other Python modules:
# COPY utils/ /app/utils/

# === (Optional) Pre-download Hugging Face models ===
# Uncomment and adapt if you want to bake models into the image.
# This increases image size but can be useful for offline use or consistent startup.
# Ensure 'transformers' and 'torch' are installed by the requirements.txt before this step.
# ENV HF_HOME=/app/huggingface_cache
# RUN mkdir -p $HF_HOME
# RUN echo "Pre-downloading model (this might take a while)..." && \
#     python -c "from transformers import AutoProcessor, LlavaForConditionalGeneration; \
#                MODEL_ID='llava-hf/llava-interleave-qwen-7b-hf'; \
#                print(f'Downloading processor for {MODEL_ID}'); \
#                AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir='$HF_HOME'); \
#                print(f'Downloading model {MODEL_ID}'); \
#                LlavaForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir='$HF_HOME'); \
#                print('Model pre-download complete.')"

# === Entrypoint ===
# Specifies the command that will be run when the container starts.
# Your script 'run_mmiu_llava_pipeline.py' will receive any arguments passed
# to 'docker run <image_name> ...' after the image name.
ENTRYPOINT ["python", "eval-runner.py"]

# === Default Command (Optional) ===
# If no command is provided to 'docker run', this will be executed.
# Often used to show help or run a default action.
# CMD ["--help"]
