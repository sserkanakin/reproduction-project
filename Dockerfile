# Torch 2.3 + CUDA 12.1 base
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git jq && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST=8.9  PYTHONUNBUFFERED=1

# Fast, ABI-matched Flash-Attention 2.6 wheel (cu121 / torch230)
RUN pip install --upgrade pip && \
    pip install flash-attn==2.5.5 bitsandbytes==0.43.1

# Clone *latest* LLaVA and install in editable mode
RUN git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /llava && \
    pip install -e /llava

# Extra deps
RUN pip install transformers==4.41.2 peft==0.10.0 accelerate==0.29.3 datasets tqdm

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
