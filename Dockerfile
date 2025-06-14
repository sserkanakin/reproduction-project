###############################################################################
#  LLaVA 1.2.2.post1  •  CUDA 12.1 • single–GPU LoRA finetune                 #
###############################################################################
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 1. system deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git build-essential curl jq ca-certificates python3 python3-pip tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. python libs (exact working pins)
RUN pip3 install --no-cache-dir \
        torch==2.2.* --extra-index-url https://download.pytorch.org/whl/cu121 \
        transformers==4.52.0 accelerate==1.7.0 peft==0.15.2 \
        bitsandbytes==0.43.1 sentencepiece einops pillow tqdm gradio \
        datasets==2.19.0

# 3. LLaVA   (contains train.py & its Trainer/ collator)
RUN git clone --depth 1 --branch v1.2.2.post1 https://github.com/haotian-liu/LLaVA.git /opt/LLaVA \
 && pip3 install /opt/LLaVA

ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1
WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash"]
