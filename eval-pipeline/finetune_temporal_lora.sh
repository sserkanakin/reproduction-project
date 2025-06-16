#!/usr/bin/env bash
set -euo pipefail

deepspeed llava/train/train_mem.py \
  --deepspeed               llava/scripts/zero3_offload.json \
  --model_name_or_path      llava-hf/llava-interleave-qwen-0.5b-hf \
  --version                 plain \
  --data_path               eval-pipeline/data/finetune_data/train.jsonl \
  --val_data_path           eval-pipeline/data/finetune_data/test.jsonl \
  --image_folder            eval-pipeline/data \
  --vision_tower            openai/clip-vit-large-patch14-336 \
  --mm_projector_type       mlp2x_gelu \
  --tune_mm_mlp_adapter     true \
  --lora_enable             true \
  --per_device_train_batch_size   4 \
  --gradient_accumulation_steps   4 \
  --num_train_epochs        1 \
  --learning_rate           5e-5 \
  --logging_steps           20 \
  --save_strategy           no \
  --bf16                    true \
  --output_dir              checkpoints/temporal_lora
