#!/usr/bin/env bash
##############################################################################
# finetune_temporal_lora.sh – single-GPU LoRA finetuning for LLaVA on an L4
# ---------------------------------------------------------------------------
# • Handles JSON/JSONL datasets.
# • Creates a tokenizer copy with pad_token=eos_token so collate_fn works.
# • Streams JSONL via --lazy_preprocess and sets max_length to 2048.
##############################################################################
set -euo pipefail

# ---------------------------- CLI & defaults --------------------------------
DATA= EVAL= IMG_ROOT= OUT=
EPOCHS=1 BATCH=4 GRAD_ACC=4 LR=5e-5
MODEL="llava-hf/llava-interleave-qwen-7b-hf"
VIT="openai/clip-vit-large-patch14-336"
TOK_DIR="/tmp/qwen_pad_tok"       # temp folder for patched tokenizer

usage() {
  echo "Usage: $0 --data TRAIN.json[l] --eval TEST.json[l] --images_root DIR --out DIR [--epochs N]" >&2
  exit 1
}

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
  jq -e '.[0].images' "$f" >/dev/null 2>&1 || \
    head -n1 "$f" | jq -e '.images' >/dev/null 2>&1 || {
    echo "ERROR: $f missing .images field" >&2; exit 1;
  }
done

# ------------- Prepare tokenizer copy with pad_token = eos ---------------
python3 <<PY
import os, sys
from transformers import AutoTokenizer
# use shell variable expansion for TOK_DIR and MODEL
TOK_DIR = os.path.expanduser("${TOK_DIR}")
MODEL = "${MODEL}"
if not os.path.isdir(TOK_DIR):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(TOK_DIR, safe_serialization=True)
    print(f"✅ tokenizer saved with pad_token → {TOK_DIR}", file=sys.stderr)
# monkey-patch AutoTokenizer to redirect MODEL to TOK_DIR
import transformers
def patched_from_pretrained(name_or_path, *args, **kwargs):
    if name_or_path == MODEL:
        return transformers.AutoTokenizer.from_pretrained(TOK_DIR, *args, **kwargs)
    return transformers.AutoTokenizer.from_pretrained.__wrapped__(name_or_path, *args, **kwargs)
transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
PY

# ---------------------- Monkey-patch pad_sequence ------------------------
python3 <<PY
import sys
import torch.nn.utils.rnn as rnn
orig_pad = rnn.pad_sequence

def pad_sequence(sequences, batch_first=False, padding_value=None):
    if padding_value is None:
        padding_value = 0.0
    return orig_pad(sequences, batch_first=batch_first, padding_value=padding_value)

rnn.pad_sequence = pad_sequence
print('✅ patched pad_sequence to default padding_value 0.0', file=sys.stderr)
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
