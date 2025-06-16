#!/usr/bin/env bash
##############################################################################
# finetune_temporal_lora.sh – L4‑ready LoRA fine‑tune for latest LLaVA (2025‑06‑16)
# • Accepts either JSONL (one‑sample‑per‑line) **or** JSON array files.
# • Auto‑installs/patches LLaVA, injects LoRA, runs BF16 training.
##############################################################################
set -euo pipefail

# ---------------------------- CLI & defaults --------------------------------
DATA= EVAL= IMG_ROOT= OUT=
EPOCHS=1 BATCH=4 GRAD_ACC=4 LR=5e-5
MODEL="llava-hf/llava-interleave-qwen-0.5b-hf"
VIT="openai/clip-vit-large-patch14-336"

usage() {
  echo "Usage: $0 --data TRAIN.json[l] --eval TEST.json[l] --images_root DIR --out DIR [--epochs N]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do case $1 in
  --data)        DATA=$2; shift 2;;
  --eval)        EVAL=$2; shift 2;;
  --images_root) IMG_ROOT=$2; shift 2;;
  --out)         OUT=$2; shift 2;;
  --epochs)      EPOCHS=$2; shift 2;;
  --batch)       BATCH=$2; shift 2;;
  --grad_acc)    GRAD_ACC=$2; shift 2;;
  --lr)          LR=$2; shift 2;;
  *) usage;;
esac; done
[[ -z $DATA || -z $EVAL || -z $IMG_ROOT || -z $OUT ]] && usage

# ---------- sanity: file exists + has .images field (JSON or JSONL) ----------
for f in "$DATA" "$EVAL"; do
  [[ ! -f $f ]] && { echo "File not found: $f" >&2; exit 1; }
  if jq -e '.[0].images' "$f" >/dev/null 2>&1; then continue; fi
  if head -n1 "$f" | jq -e '.images' >/dev/null 2>&1; then continue; fi
  echo "ERROR: $f is neither JSON‑array nor JSONL with .images field" >&2; exit 1;
done

# -------------------------- Install LLaVA & patch it ------------------------
python3 - <<'PY'
import importlib, sys
tok_mod = importlib.import_module('llava.train.train').Tokenizer
tokenizer = tok_mod.get_tokenizer(model_name="llava-hf/llava-interleave-qwen-7b-hf")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token          # id = 0
    tokenizer.save_pretrained("/tmp/llava_tmp_tok")    # saved once
    print("✅ pad_token set to eos_token (id 0)", file=sys.stderr)
PY

# ------------------------------- Training -----------------------------------
python3 -m llava.train.train_mem \
  --model_name_or_path            "$MODEL" \
  --version                       plain \
  --data_path                     "$DATA" \
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
  --model_max_length              2048 \
  --lazy_preprocess               true \
  --output_dir                    "$OUT"
