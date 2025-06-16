#!/usr/bin/env bash
##############################################################################
# run_temporal_lora.sh
#
# LoRA-fine-tune `llava-hf/llava-interleave-qwen-7b-hf`
# on the multi-image temporal-ordering dataset created with
# `prepare_temporal_dataset.py`, using **any** recent LLaVA install.
#
# It first injects the missing symbol `LlavaLlamaForCausalLM`
# into `llava.model` when newer commits no longer export it.
##############################################################################

set -euo pipefail

# ---------- user args ----------
EPOCHS=3 BATCH=4 GRAD_ACC=4 LR=5e-5
DATA= EVAL= IMG_ROOT= OUT=
MODEL="llava-hf/llava-interleave-qwen-7b-hf"
VISION="openai/clip-vit-large-patch14-336"

usage() { echo "Usage: $0 --data train.jsonl --eval test.jsonl --images_root DIR --out OUT_DIR [--epochs N]" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --data) DATA=$2; shift 2;;
    --eval) EVAL=$2; shift 2;;
    --images_root) IMG_ROOT=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --epochs) EPOCHS=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --grad_acc) GRAD_ACC=$2; shift 2;;
    --lr) LR=$2; shift 2;;
    *) usage;;
  esac
done
[[ -z $DATA || -z $EVAL || -z $IMG_ROOT || -z $OUT ]] && usage

# ---------- sanity ----------
for f in "$DATA" "$EVAL"; do
  [[ ! -f $f ]] && { echo "File not found: $f" >&2; exit 1; }
  head -n1 "$f" | jq -e '.images' >/dev/null 2>&1 \
    || { echo "ERROR: $f not LLaVA-JSONL" >&2; exit 1; }
done

# ---------- HOT-PATCH ----------
python - <<'PY'
"""
Patch newer LLaVA builds so that `from llava.model import LlavaLlamaForCausalLM`
works again (needed by train_mem.py).
"""
import importlib, sys, types

try:
    import llava.model as _root
    import llava.model.language_model.llava_llama as _llm
    setattr(_root, "LlavaLlamaForCausalLM", _llm.LlavaLlamaForCausalLM)
except Exception as e:
    print("⚠️  Hot-patch skipped:", e, file=sys.stderr)
PY

# ---------- train ----------
python -m llava.train.train_mem \
  --model_name_or_path            "$MODEL" \
  --version                       plain \
  --data_path                     "$DATA" \
  --val_data_path                 "$EVAL" \
  --image_folder                  "$IMG_ROOT" \
  --vision_tower                  "$VISION" \
  --mm_projector_type             mlp2x_gelu \
  --tune_mm_mlp_adapter           true \
  --lora_enable                   true \
  --lora_r                        64 \
  --lora_alpha                    16 \
  --lora_dropout                  0.05 \
  --per_device_train_batch_size   $BATCH \
  --per_device_eval_batch_size    $BATCH \
  --gradient_accumulation_steps   $GRAD_ACC \
  --num_train_epochs              $EPOCHS \
  --learning_rate                 $LR \
  --warmup_ratio                  0.03 \
  --lr_scheduler_type             cosine \
  --save_strategy                 steps \
  --save_steps                    200 \
  --logging_steps                 20 \
  --model_max_length              8192 \
  --fp16                          true \
  --output_dir                    "$OUT"
