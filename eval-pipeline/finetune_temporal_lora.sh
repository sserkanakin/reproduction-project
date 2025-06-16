#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Robust LoRA fine-tune launcher for LLaVA                                    #
# Works with *any* commit of LLaVA because it builds a shim if needed.        #
##############################################################################

# ---------------- user args ----------------
DATA= EVAL= IMG_ROOT= OUT=
EPOCHS=3 BATCH=4 GRAD_ACC=4 LR=5e-5
MODEL="llava-hf/llava-interleave-qwen-7b-hf"
VISION="openai/clip-vit-large-patch14-336"
DO_INSTALL=true   # set false with --no-install

usage() {
  cat <<EOF
Usage: $0 --data train.jsonl --eval test.jsonl --images_root DIR --out OUT_DIR
         [--epochs N] [--batch N] [--grad_acc N] [--lr LR] [--no-install]
EOF
  exit 1
}

# -------------- parse cli -----------------
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
    --no-install)  DO_INSTALL=false; shift 1;;
    *) usage;;
  esac
done
[[ -z $DATA || -z $EVAL || -z $IMG_ROOT || -z $OUT ]] && usage

# -------------- sanity ---------------------
for f in "$DATA" "$EVAL"; do
  [[ ! -f "$f" ]] && { echo "File not found: $f" >&2; exit 1; }
  head -n1 "$f" | jq -e '.images' >/dev/null 2>&1 \
    || { echo "ERROR: $f not LLaVA-JSONL" >&2; exit 1; }
done

# -------------- ensure llava --------------
python - <<PY
import importlib, subprocess, sys, pkgutil, types, inspect, os, textwrap

if $DO_INSTALL and importlib.util.find_spec("llava") is None:
    print("⏳  pip-installing latest LLaVA …", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "git+https://github.com/haotian-liu/LLaVA.git@main"])

try:
    import llava  # may still raise ImportError inside __init__
except Exception as exc:
    print("⚠️  llava import failed → building shim:", exc, file=sys.stderr)
    # Load only the deep module that contains the class
    sm = importlib.import_module("llava.model.language_model.llava_llama")
    shim = types.ModuleType("llava")
    shim.model = types.ModuleType("llava.model")
    shim.model.LlavaLlamaForCausalLM = sm.LlavaLlamaForCausalLM
    sys.modules["llava"] = shim
    sys.modules["llava.model"] = shim.model
    print("✅  llava shim created", file=sys.stderr)
else:
    # If import succeeded but export missing, patch it.
    try:
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        import llava.model as _root
        if not hasattr(_root, "LlavaLlamaForCausalLM"):
            setattr(_root, "LlavaLlamaForCausalLM", LlavaLlamaForCausalLM)
            print("✅  Patched LlavaLlamaForCausalLM into llava.model", file=sys.stderr)
    except Exception as e:
        print("⚠️  Patched failed:", e, file=sys.stderr)
PY

# -------------- train ----------------------
python -m llava.train.train_mem \
  --model_name_or_path            "$MODEL" \
  --version                       plain \
  --data_path                     "$DATA" \
  --val_data_path                 "$EVAL" \
  --image_folder                  "$IMG_ROOT" \
  --vision_tower                  "$VISION" \
  --mm_projector_type             mlp2x_gelu \
  --tune_mm_mlp_adapter           True \
  --lora_enable                   True \
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
  --fp16                          True \
  --output_dir                    "$OUT"
