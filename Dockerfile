#### to build: docker build -t llava:latest .
#### to run: docker run --gpus all -it --rm -v $(pwd):/workspace llava:latest
#FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
#
#RUN apt-get update && apt-get install -y git jq ninja-build && rm -rf /var/lib/apt/lists/*
#
#ENV CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
#    TORCH_CUDA_ARCH_LIST=8.9 \
#    PYTHONUNBUFFERED=1
#
#
#
#
#
## 2️⃣  BitsAndBytes for 4-bit LoRA
#RUN pip install bitsandbytes==0.43.1
#
## 3️⃣  Latest LLaVA
#RUN pip install "git+https://github.com/haotian-liu/LLaVA.git@main"
#
## 4️⃣  Helpers
#RUN pip install transformers==4.40.0 peft==0.10.0 accelerate==0.29.3 datasets tqdm deepspeed openai dotenv
#
#
## 1️⃣  Build flash-attn from source (10 min on L4)
#RUN pip install --upgrade pip && \
#    pip install --no-build-isolation flash-attn==2.5.9.post1 \
#        --extra-index-url https://pypi.org/simple
#
#RUN pip uninstall -y flash_attn flash_attn_2_cuda && \
#     pip install flash-attn==2.5.5
#
#WORKDIR /workspace
#ENTRYPOINT ["/bin/bash"]
# -- Base Image --
# We select a base image from NVIDIA that matches your VM's key specs:
# CUDA 12.1.1 (compatible with host's 12.1), Python 3.10, on Debian 11.
# The 'devel' tag includes the CUDA compiler (nvcc), which is needed for some packages.
ARG CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.10
ARG OS=debian11
FROM nvidia/cuda:${CUDA_VERSION}-devel-python${PYTHON_VERSION}-${OS}

# -- Metadata --
LABEL maintainer="Serkan Akin"
LABEL description="Dockerfile to finetune llava-interleave on a custom dataset."

# -- Environment Setup --
# Prevents interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set the working directory inside the container
WORKDIR /app

# -- System Dependencies --
# Install git to clone the repository and build-essential for any C++ extensions.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -- Application Code --
# Clone the finetuning repository directly into our working directory.
RUN git clone https://github.com/zjysteven/lmms-finetune.git .

# -- Python Dependencies --
# Upgrade pip and install all required packages from the repository's requirements file.
# Using --no-cache-dir keeps the image layer smaller.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -- Entrypoint --
# The default command when the container starts.
# We will drop into a bash shell to allow for interactive use (e.g., editing a script before running).
CMD ["bash"]