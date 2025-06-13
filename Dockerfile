##############################################################################
# LLaVA-Interleave-Qwen-7B fine-tune container – CUDA 12.1  (Debian-11 host)
##############################################################################
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ── 1.  OS utilities ────────────────────────────────────────────────────────
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git curl jq tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["/usr/bin/tini", "--"]

# ── 2.  Python deps (pinned) ────────────────────────────────────────────────
RUN pip install --no-cache-dir \
      transformers==4.41.0 \
      trl==0.9.0 \
      accelerate==0.28.0 \
      peft==0.10.0 \
      bitsandbytes==0.43.1 \
      deepspeed==0.14.2 \
      datasets==2.19.0 \
      Pillow einops sentencepiece tqdm python-dotenv \
      openai>=1.25.0

# If you keep a root-level requirements.txt **and** want extras installed,
# uncomment the next two lines:
# COPY requirements.txt /tmp/req.txt
# RUN pip install --no-cache-dir -r /tmp/req.txt

# ── 3.  Workspace / caches ─────────────────────────────────────────────────
WORKDIR /workspace
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/transformers \
    PYTHONUNBUFFERED=1 \
    NCCL_P2P_DISABLE=1        # avoids an occasional NCCL P2P bug on GCP

# ── 4.  Copy project code ──────────────────────────────────────────────────
# Everything under eval-pipeline/ ends up at /workspace/eval-pipeline/
COPY eval-pipeline/ /workspace/eval-pipeline/
# run-inference (if you want it inside, too):
COPY run-inference.py /workspace/run-inference.py

# convenience: add scripts dir to PATH
ENV PATH="/workspace/eval-pipeline:${PATH}"

CMD ["bash"]
