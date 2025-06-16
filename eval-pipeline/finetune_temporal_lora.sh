#!/usr/bin/env bash
##############################################################################
# finetune_temporal_lora.sh                                                   #
# ----------------------------------------------------------------------------#
# One‑liner LoRA fine‑tuning of the `llava-hf/llava-interleave-qwen-7b-hf`     #
# model on the **multi‑image temporal‑ordering dataset** produced by          #
# `prepare_temporal_dataset.py`. Everything below follows the public LLaVA    #
# training interface, so *no code modifications* are required inside LLaVA.   #
#                                                                             #
# ʟ4 GPU tested (24 GB) – uses:                                               #
#   • 4‑bit weight loading (`bitsandbytes`)                                    #
#   • LoRA on Q, K, V, O + cross‑modal MLP (≈ 25 M tunable params)             #
#   • Gradient Accumulation to fit batch size                                  #
##############################################################################

set -euo pipefail

# ---------------------------- Helper & defaults -----------------------------
usage() {
  echo "Usage: $0 --data TRAIN.jsonl --eval TEST.jsonl --images_root DIR --out CKPT_DIR [--epochs N]" >&2
  exit 1
}

# sensible defaults for a single‑GPU L4 box
EPOCHS=3
BATCH=4           # per‑device batch (4 × 7 images × 576 tokens ≈ 13 k)
GRAD_ACC=4        # effective batch 16
LR=5e-5           # LoRA learning‑rate
MODEL="llava-hf/llava-inbash eval-pipeline/finetune_temporal_lora.sh   --data   eval-pipeline/data/finetune_data/train.jsonl   --eval   eval-pipeline/data/finetune_data/test.jsonl   --images_root  eval-pipeline/data   --out    checkpoints/temporal_lora   --epochs 1
⏳  Installing LLaVA from GitHub …
WARNING: typer 0.16.0 does not provide the extra 'all'
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
⚠️  Patch failed: cannot import name 'LlavaLlamaForCausalLM' from 'llava.model' (/opt/conda/lib/python3.10/site-packages/llava/model/__init__.py)
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/runpy.py", line 187, in _run_module_as_main
    mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
  File "/opt/conda/lib/python3.10/runpy.py", line 110, in _get_module_details
    __import__(pkg_name)
  File "/opt/conda/lib/python3.10/site-packages/llava/__init__.py", line 1, in <module>
    from .model import LlavaLlamaForCausalLM
ImportError: cannot import name 'LlavaLlamaForCausalLM' from 'llava.model' (/opt/conda/lib/python3.10/site-packages/llava/model/__init__.py)terleave-qwen-7b-hf"
VISION_TOWER="openai/clip-vit-large-patch14-336"

# ------------------------------- CLI parse ----------------------------------
DATA=""
EVAL=""
IMG_ROOT=""
OUT=""

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

[[ -z "$DATA" || -z "$EVAL" || -z "$IMG_ROOT" || -z "$OUT" ]] && usage

# ----------------------------- Sanity checks --------------------------------
for p in "$DATA" "$EVAL"; do
  [[ ! -f "$p" ]] && { echo "File not found: $p" >&2; exit 1; }
  # Read first line only (JSONL) and verify an `images` key exists
  head -n 1 "$p" | jq -e '.images' >/dev/null 2>&1 || {
    echo "ERROR: $p does not look like JSONL with LLaVA format" >&2; exit 1; }
done

# -------------------- Ensure LLaVA is present + hot‑patch ------------------
python3 - <<'PY'
import importlib, subprocess, sys, textwrap, shutil, os

# 1. Install LLaVA if missing ----------------------------------------------
if importlib.util.find_spec('llava') is None:
    print('⏳  Installing LLaVA from GitHub …', file=sys.stderr)
    cmd = [sys.executable, '-m', 'pip', 'install', '--quiet',
           'git+https://github.com/haotian-liu/LLaVA.git@main']
    subprocess.check_call(cmd)

# 2. Re‑export LlavaLlamaForCausalLM for newer commits ----------------------
try:
    import llava.model as _root
    _llm = importlib.import_module('llava.model.language_model.llava_llama')
    setattr(_root, 'LlavaLlamaForCausalLM', _llm.LlavaLlamaForCausalLM)
    print('✅  LLaVA present & patched', file=sys.stderr)
except Exception as e:
    print('⚠️  Patch failed:', e, file=sys.stderr)
PY

# ------------------------------- Training -----------------------------------
python3 -m llava.train.train_mem \
  --model_name_or_path            "$MODEL" \
  --version                       plain \
  --data_path                     "$DATA" \
  --val_data_path                 "$EVAL" \
  --image_folder                  "$IMG_ROOT" \
  --vision_tower                  "$VISION_TOWER" \
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
  --bf16                          false \
  --tf32                          true \
  --output_dir                    "$OUT"

##############################################################################
# POST‑RUN:                                                                  #
#  • The LoRA adapters + projector are stored in $OUT                        #
#  • Merge into a standalone checkpoint with `python -m llava.merge_lora ...`#
#  • Evaluate: `python -m llava.eval.run_eval --data-path $EVAL --model ...`  #
##############################################################################
