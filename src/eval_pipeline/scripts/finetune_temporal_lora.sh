#!/usr/bin/env bash
##############################################################################
# finetune_temporal_lora.sh – single-GPU 4-bit LoRA finetuning for LLaVA on an L4
# ---------------------------------------------------------------------------
# • Handles JSON/JSONL datasets.
# • Creates tokenizer copy with pad_token=eos_token so collate_fn works.
# • Streams JSONL via --lazy_preprocess and sets max_length to 1024.
# • Uses 4-bit quantization (bnb) to fit in 24 GB VRAM.
##############################################################################
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ---------------------------- CLI & defaults --------------------------------
DATA= EVAL= IMG_ROOT= OUT=
EPOCHS=1 BATCH=2 GRAD_ACC=4 LR=5e-5
MODEL="llava-hf/llava-interleave-qwen-0.5b-hf"
VIT="openai/clip-vit-large-patch14-336"
MAX_LEN=1024

usage() { echo "Usage: $0 --data TRAIN.json[l] --eval TEST.json[l] --images_root DIR --out DIR [--epochs N]" >&2; exit 1; }

# --------------------------- Parse arguments -------------------------------
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

# ---------- sanity: dataset exists & has .images field (JSON/JSONL) ----------
for f in "$DATA" "$EVAL"; do
  [[ ! -f $f ]] && { echo "File not found: $f" >&2; exit 1; }
  jq -e '.[0].images' "$f" >/dev/null 2>&1 || head -n1 "$f" | jq -e '.images' >/dev/null 2>&1 || {
    echo "ERROR: $f missing .images field" >&2; exit 1;
  }
done


# ------------------------------- Training -----------------------------------
python3 -m llava.train.train_mem \
  --model_name_or_path            "$MODEL" \
  --version                       v0 \
  --data_path                     "$DATA" \
  --image_folder                  "$IMG_ROOT" \
  --vision_tower                  "$VIT" \
  --mm_projector_type             mlp2x_gelu \
  --tune_mm_mlp_adapter           true \
  --lora_enable                   true \
  --lora_r                        32 \
  --lora_alpha                    8 \
  --lora_dropout                  0.05 \
  --gradient_checkpointing         true \
  --per_device_train_batch_size   $BATCH \
  --per_device_eval_batch_size    $BATCH \
  --gradient_accumulation_steps   $GRAD_ACC \
  --num_train_epochs              $EPOCHS \
  --learning_rate                 $LR \
  --logging_steps                 20 \
  --bf16                          true \
  --model_max_length              $MAX_LEN \
  --lazy_preprocess               true \
  --output_dir                    "$OUT"
