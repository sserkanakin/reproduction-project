#!/usr/bin/env bash
set -euo pipefail

# -- CONFIGURATION ────────────────────────────────────────────────────────────
MODEL="llava-hf/llava-interleave-qwen-0.5b-hf"
VERSION="llava_v1"
TRAIN_JSON="eval-pipeline/data/finetune_data/train.json"
#EVAL_JSON="eval-pipeline/data/finetune_data/test.jsonl"
IMG_ROOT="eval-pipeline/data"
PRETRAIN_PROJECTOR="checkpoints/llava-0.5b-pretrain/mm_projector.bin"
ZERO3_CFG="scripts/zero3.json"
OUTDIR="checkpoints/temporal_lora_deepspeed"
EPOCHS=3
# LoRA hyperparameters
LORA_R=64
LORA_ALPHA=16
MM_PROJECTOR_LR=2e-5
# Train batch = per_device × grad_accum_steps × #gpus  →  adjust to your L4
BATCH=2
GRAD_ACC=4
LR=2e-5

# -- FINETUNE ─────────────────────────────────────────────────────────────────
deepspeed LLaVa/llava/train/train_mem.py \
  --deepspeed              $ZERO3_CFG \
  --lora_enable            True \
  --lora_r                 $LORA_R \
  --lora_alpha             $LORA_ALPHA \
  --mm_projector_lr        $MM_PROJECTOR_LR \
  --pretrain_mm_mlp_adapter $PRETRAIN_PROJECTOR \
  --model_name_or_path     $MODEL \
  --version                $VERSION \
  --data_path              $TRAIN_JSON \
  --image_folder           $IMG_ROOT \
  --vision_tower           openai/clip-vit-large-patch14-336 \
  --mm_projector_type      mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end    False \
  --mm_use_im_patch_token  False \
  --image_aspect_ratio     pad \
  --group_by_modality_length True \
  --bf16                   True \
  --gradient_checkpointing True \
  --lazy_preprocess        True \
  --num_train_epochs       $EPOCHS \
  --per_device_train_batch_size  $BATCH \
  --per_device_eval_batch_size   $BATCH \
  --gradient_accumulation_steps  $GRAD_ACC \
  --learning_rate          $LR \
  --weight_decay           0.0 \
  --warmup_ratio           0.03 \
  --lr_scheduler_type      cosine \
  --logging_steps          20 \
  --save_strategy          steps \
  --save_steps             200 \
  --save_total_limit       1 \
  --tf32                   True \
  --model_max_length       2048 \
  --dataloader_num_workers 4 \
  --output_dir             $OUTDIR
