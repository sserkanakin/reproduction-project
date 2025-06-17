#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIGURATION (edit if needed) ──────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-checkpoints/llava-merged}"                 # your merged LoRA checkpoint
MODEL_BASE="${MODEL_BASE:-llava-hf/llava-interleave-qwen-0.5b-hf}"  # base model for tokenizer & vision

# ─── USAGE / ARGS PARSING ───────────────────────────────────────────────────
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <img1>[,<img2>,...] <\"YOUR QUESTION\"> [options]"
  echo
  echo "Positional:"
  echo "  <image-file>   comma-sep list of image paths"
  echo "  <query>        your question (wrap in quotes)"
  echo
  echo "Options:"
  echo "  --conv-mode MODE       (e.g. llava_v1)       [auto-detected if omitted]"
  echo "  --sep SEPCHAR          (default: , )"
  echo "  --temperature FLOAT    (default: 0    greedy)"
  echo "  --top_p FLOAT          (default: 1.0)"
  echo "  --num_beams INT        (default: 1)"
  echo "  --max_new_tokens INT   (default: 256)"
  exit 1
fi

IMAGE_FILE="$1"; shift
QUERY="$1";      shift

# defaults
CONV_MODE=""
SEP=","
TEMPERATURE="0"
TOP_P="1.0"
NUM_BEAMS="1"
MAX_NEW_TOKENS="256"

# parse named flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --conv-mode)       CONV_MODE="$2";       shift 2;;
    --sep)             SEP="$2";             shift 2;;
    --temperature)     TEMPERATURE="$2";     shift 2;;
    --top_p)           TOP_P="$2";           shift 2;;
    --num_beams)       NUM_BEAMS="$2";       shift 2;;
    --max_new_tokens)  MAX_NEW_TOKENS="$2";  shift 2;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ─── INVOKE THE OFFICIAL EVAL SCRIPT ──────────────────────────────────────────
python LLaVA/llava/eval/run_llava.py \
  --model-path    "$MODEL_PATH" \
  --model-base    "$MODEL_BASE" \
  --image-file    "$IMAGE_FILE" \
  --query         "$QUERY" \
  $( [ -n "$CONV_MODE" ] && echo "--conv-mode $CONV_MODE" ) \
  --sep           "$SEP" \
  --temperature   "$TEMPERATURE" \
  --top_p         "$TOP_P" \
  --num_beams     "$NUM_BEAMS" \
  --max_new_tokens "$MAX_NEW_TOKENS"
