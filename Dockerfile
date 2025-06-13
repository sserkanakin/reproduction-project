##############################################################################
# LLaVA-Interleave-Qwen-7B fine-tuning image – CUDA 12.1, PyTorch 2.2, Debian-11 host
##############################################################################
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ---------- OS utilities ----------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git curl jq tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Make Ctrl-C propagate properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# ---------- Python dependencies ----------
# Pinned versions verified in the earlier instructions
RUN pip install --no-cache-dir \
      transformers==4.41.0 \
      trl==0.9.0 \
      accelerate==0.28.0 \
      peft==0.10.0 \
      bitsandbytes==0.43.1 \
      deepspeed==0.14.2 \
      datasets==2.19.0 \
      sentencepiece einops Pillow tqdm python-dotenv \
      openai>=1.25.0

# ---------- Set workdir & caches ----------
WORKDIR /workspace
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/transformers \
    PYTHONUNBUFFERED=1 \
    NCCL_P2P_DISABLE=1

# ---------- Copy project scripts ----------
#   host:  project/scripts/prepare_temporal_dataset.py
#   host:  project/scripts/run_trl_ft.py
COPY scripts/ /workspace/scripts/

# Make them executable for convenience
RUN chmod +x /workspace/scripts/*.py

# Add scripts to PATH so we can call them as entrypoints
ENV PATH="/workspace/scripts:${PATH}"

# ---------- Default command ----------
CMD ["bash"]
