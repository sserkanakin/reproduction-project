#!/usr/bin/env python
"""run_trl_ft.py ── LoRA fine‑tune for LLaVA‑Interleave‑Qwen‑7B

Assumptions / defaults
======================
* All code runs **inside the Docker container** we built earlier.
* Images live at           `/workspace/eval-pipeline/data/Continuous-temporal/...`
* Training JSONL lives at   `/workspace/eval-pipeline/data/finetuning_data/llava_temporal_train.jsonl`
* Optional dev JSONL lives  `/workspace/eval-pipeline/data/finetuning_data/llava_temporal_dev.jsonl`

Override anything via environment variables or CLI flags (see `argparse`).

Launch examples
---------------
Single‑GPU (A100‑80 or L4):
```bash
docker run --gpus all --rm -it \
  -v $PWD/eval-pipeline:/workspace/eval-pipeline \
  llava-temporal:latest \
  python /workspace/eval-pipeline/run_trl_ft.py
```

Multi‑GPU with DeepSpeed ZeRO‑2 (4×A100‑40 GB):
```bash
export NUM_GPUS=4
mkdir -p eval-pipeline/outputs
accelerate launch \
  --multi_gpu --num_processes $NUM_GPUS --mixed_precision bf16 \
  --deepspeed_config_file /workspace/eval-pipeline/ds_zero2.json \
  /workspace/eval-pipeline/run_trl_ft.py
```

The script writes its PEFT adapter to `outputs/temporal_lora/` and evaluation
metrics to TensorBoard logs under the same directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset, disable_caching
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer

disable_caching()  # avoids clutter in large trainings

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_id", default="llava-hf/llava-interleave-qwen-7b-hf")
    p.add_argument("--image_root", default="/workspace/eval-pipeline/data")
    p.add_argument("--train_jsonl", default="/workspace/eval-pipeline/data/finetuning_data/llava_temporal_train.jsonl")
    p.add_argument("--eval_jsonl",  default="/workspace/eval-pipeline/data/finetuning_data/llava_temporal_dev.jsonl")

    p.add_argument("--output_dir",  default="/workspace/eval-pipeline/outputs/temporal_lora")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    return p.parse_args()

# ---------- Preprocess function ----------

def make_preprocess(processor: LlavaProcessor, image_root: Path):
    from PIL import Image

    def _inner(example: Dict):
        img_paths = [image_root / p for p in example["images"]]
        images = [Image.open(p) for p in img_paths]
        prompt = processor.apply_chat_template(example["conversations"], add_generation_prompt=False)
        model_inputs = processor(text=prompt, images=images, return_tensors="pt")
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

    return _inner

# ---------- Main ----------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading processor & model…")
    processor = LlavaProcessor.from_pretrained(args.model_id)

    # 4‑bit QLoRA config (bnb 8‑bit for linear, 4‑bit for base weights)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    # PEFT LoRA
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    print("[INFO] Loading dataset…")
    files = {"train": args.train_jsonl}
    if Path(args.eval_jsonl).exists():
        files["eval"] = args.eval_jsonl
    ds = load_dataset("json", data_files=files)

    preprocess_fn = make_preprocess(processor, Path(args.image_root))
    ds = ds.map(preprocess_fn, remove_columns=ds["train"].column_names, num_proc=8)

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        dataset_text_field=None,
        peft_config=peft_cfg,
        max_seq_length=args.max_seq_length,
        generation_kwargs=dict(max_new_tokens=64),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        output_dir=args.output_dir,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
    )

    print("[INFO] Starting training…")
    trainer.train()

    print("[INFO] Saving PEFT adapter & tokenizer…")
    trainer.save_model(args.output_dir)        # adapter & config
    processor.save_pretrained(args.output_dir) # tokenizer + image processors
    print("[INFO] Training complete. Adapter saved in", args.output_dir)


if __name__ == "__main__":
    main()
