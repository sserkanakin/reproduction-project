# CUDA‑enabled base image with PyTorch 2.3 (CUDA 12.1) – perfectly matches NVIDIA L4 (sm_89)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

###############################################################################
# System deps
###############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential cmake \
        libjpeg-dev libpng-dev jq && \
    rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=8.9

###############################################################################
# Python deps – keep this layer stable so you **don’t** need to rebuild when
# you tweak local scripts. Only re‑build if you change these versions.
###############################################################################

# Copy *only* requirements.txt (enables Docker cache).  None of your project
# code is copied, so subsequent edits don’t invalidate the image – just mount
# the repo at runtime.
COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    # Flash‑Attention 2.5.5 (compiled for sm_89) – provides efficient MHA
    pip install --no-build-isolation flash-attn==2.5.5 && \
    # bitsandbytes 0.43.1 – 4‑bit quantisation for QLoRA
    pip install --no-build-isolation bitsandbytes==0.43.1 && \
    # Project‑specific Python deps
    pip install --requirement /tmp/requirements.txt && \
    # LLaVA (main branch) – includes llava.train.train_mem
    pip install git+https://github.com/haotian-liu/LLaVA.git@main && \
    rm /tmp/requirements.txt

###############################################################################
# Runtime
###############################################################################
WORKDIR /workspace

# All scripts + data stay on the *host* and are bind‑mounted, so there’s no
# need to copy them here.  Example:
#   docker run --gpus all -it -v $(pwd):/workspace llava-temporal
# Now `/workspace` inside the container shows your live repo, including:
#   Dockerfile  README.md  eval-pipeline/  requirements.txt  run-inference.py
#   … plus the new finetune_data you generate.
###############################################################################

ENTRYPOINT ["/bin/bash"]
