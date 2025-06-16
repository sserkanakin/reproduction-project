FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git jq && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST=8.9  \
    PYTHONUNBUFFERED=1

# ABI-matched Flash-Attention & BitsAndBytes
RUN pip install --upgrade pip && \
    pip install flash-attn==2.6.0 bitsandbytes==0.43.1

# latest LLaVA (main)
RUN pip install "git+https://github.com/haotian-liu/LLaVA.git@main"

# helpers
RUN pip install transformers==4.41.2 peft==0.10.0 accelerate==0.29.3 \
           datasets tqdm

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
