###############################################################################
#  LLaVA-Interleave-Qwen-7B LoRA fine-tune  –  single NVIDIA L4 (CUDA 12.1)   #
#  Debian-compatible image, model cached, code bind-mounted at runtime        #
###############################################################################

# 1. CUDA 12.1 runtime (Ubuntu 22.04, Debian-compatible)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 2. System + Python deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git curl jq ca-certificates tini \
        python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Python libraries (pins match run_trl_ft.py)
RUN pip3 install --no-cache-dir \
        torch==2.2.* --extra-index-url https://download.pytorch.org/whl/cu121 \
        transformers==4.45.1 \
        trl==0.9.6 \
        accelerate==0.28.0 \
        peft==0.10.0 \
        bitsandbytes==0.43.1 \
        datasets==2.19.0 \
        pillow einops sentencepiece tqdm python-dotenv tensorboard \
        openai>=1.25.0

RUN git clone --depth 1 --branch v1.6.4 https://github.com/haotian-liu/LLaVA.git

# 4. ENV + workspace
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    NCCL_P2P_DISABLE=1
WORKDIR /workspace

# 5. Pre-download the 7-B model (≈13 GB, done once at build time)
RUN python3 - <<'PY'
from transformers import AutoProcessor, LlavaForConditionalGeneration
model_id = "llava-hf/llava-interleave-qwen-7b-hf"
print(f"↓ Caching {model_id} …")
AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
LlavaForConditionalGeneration.from_pretrained(model_id, device_map={"": "meta"})
print("✓ Model cached in image layer")
PY

# 6. Tiny entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
