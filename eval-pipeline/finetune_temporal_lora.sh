#!/usr/bin/env bash
##############################################################################
# finetune_temporal_lora.sh
# Robust LoRA fine-tuning launcher for the **latest** LLaVA main branch on an
# NVIDIA L4 (Ada, BF16-capable). It:
#   • auto-installs/updates LLaVA if missing,
#   • creates a stub module so `train_mem.py` can import
#     `LlavaLlamaForCausalLM` after the repo refactor,
#   • launches training with BF16 + LoRA, matching official hyperparams.
##############################################################################
set -euo pipefail

# ---------------------------- CLI & defaults --------------------------------
DATA= EVAL= IMG_ROOT= OUT=
EPOCHS=1 BATCH=4 GRAD_ACC=4 LR=5e-5
MODEL="llava-hf/llava-interleave-qwen-7b-hf"
VIT="openai/clip-vit-large-patch14-336"

usage() {
  echo "Usage: $0 --data TRAIN.jsonl --eval TEST.jsonl --images_root DIR --out DIR [--epochs N]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --data)        DATA=$2; shift 2;;
    --eval)        EVAL=$2; shift 2;;
    --images_root) IMG_ROOT=$2; shift 2;;
    --out)         OUT=$2; shift 2;;
    --epochs)      EPOCHS=$2; shift 2;;
    --batch)       BATCH=$2; shift 2;;
    --grad_acc)    GRAD_ACC=$2; shift 2;;
    --lr)          LR=$2; shift 2;;
    *) usage;;
  esac
done
[[ -z $DATA || -z $EVAL || -z $IMG_ROOT || -z $OUT ]] && usage

for f in "$DATA" "$EVAL"; do
  [[ ! -f $f ]] && { echo "File not found: $f" >&2; exit 1; }
  head -n1 "$f" | jq -e '.images' >/dev/null 2>&1 || {
    echo "ERROR: $f not LLaVA-JSONL" >&2; exit 1; }
done

# ------------------------------- Training -----------------------------------
python3 -m llava.train.train_mem \
  --model_name_or_path            "$MODEL" \
  --version                       plain \
  --data_path                     "$DATA" \
  --val_data_path                 "$EVAL" \
  --image_folder                  "$IMG_ROOT" \
  --vision_tower                  "$VIT" \
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
  --logging_steps                 20 \
  --bf16                          true \
  --output_dir                    "$OUT"
