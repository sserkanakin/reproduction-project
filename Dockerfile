# ──────────────────────────────────────────────────────────────────────────────
#  Base image: Google Deep Learning VM CUDA 12.1 (Debian 11, Python 3.10)
# ──────────────────────────────────────────────────────────────────────────────
FROM us-docker.pkg.dev/deeplearning-platform-release/gcp-deeplearning/common-cu121:latest

# 1. System build tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl jq tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python deps (same pins as the script)
RUN pip install --no-cache-dir \
        transformers==4.45.1 \
        trl==0.9.6 \
        accelerate==0.28.0 \
        peft==0.10.0 \
        bitsandbytes==0.43.1 \
        datasets==2.19.0 \
        Pillow einops sentencepiece tqdm python-dotenv \
        openai>=1.25.0 tensorboard

# 3. Workspace & HF caches
WORKDIR /workspace
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    NCCL_P2P_DISABLE=1

# 4. **Pre-download the LLaVA Interleave base model**  (~13 GB once)
RUN python - <<'PY'
from transformers import AutoProcessor, LlavaForConditionalGeneration
model_id = "llava-hf/llava-interleave-qwen-7b-hf"
print("⏬  Downloading", model_id)
AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
LlavaForConditionalGeneration.from_pretrained(model_id)
print("✅  Model cached inside the image")
PY

# NOTE: We do **NOT** copy your project code!  You’ll mount it at runtime:
#   -v $PWD/eval-pipeline:/workspace/eval-pipeline
#
# That way, any edits to run_trl_ft.py or data scripts are picked up instantly
# without rebuilding the image.

# 5. Default command (rarely used because we pass an explicit python file)
CMD ["bash"]
