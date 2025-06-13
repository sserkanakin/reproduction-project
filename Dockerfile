###############################################################################
#  LLaVA-Interleave-Qwen-7B LoRA fine-tune  –  single NVIDIA L4 (CUDA 12.1)   #
#  Debian base, model baked into the image, code bind-mounted at runtime      #
###############################################################################

# ── 1. Base: Debian 11 + CUDA 12.1 runtime + Python 3.10 ─────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── 2. System deps ───────────────────────────────────────────────────────────
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git curl jq ca-certificates tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── 3. Python libs (pins match run_trl_ft.py) ────────────────────────────────
RUN pip install --no-cache-dir \
        torch==2.2.* --extra-index-url https://download.pytorch.org/whl/cu121 \
        transformers==4.45.1 \
        trl==0.9.6 \
        accelerate==0.28.0 \
        peft==0.10.0 \
        bitsandbytes==0.43.1 \
        datasets==2.19.0 \
        pillow einops sentencepiece tqdm python-dotenv tensorboard \
        openai>=1.25.0

# ── 4. ENV + workspace ───────────────────────────────────────────────────────
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    NCCL_P2P_DISABLE=1
WORKDIR /workspace

RUN echo "from transformers import AutoProcessor, LlavaForConditionalGeneration
model_id = 'llava-hf/llava-interleave-qwen-7b-hf'
print(f'↓ Caching {model_id} in image …')
AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
LlavaForConditionalGeneration.from_pretrained(model_id, device_map={'': 'meta'})
print('✓ Model cached')
" | python -

# ── 6. Tiny entrypoint so container stays PID 1-clean ────────────────────────
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
