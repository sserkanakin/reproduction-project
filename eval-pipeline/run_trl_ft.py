#!/usr/bin/env python
"""run_trl_ft.py ── Single‑GPU (L4, 24 GB) QLoRA fine‑tune for
LLaVA‑Interleave‑Qwen‑7B

* 4‑bit NF4 QLoRA, compute in **fp16** (L4 lacks BF16).
* Batch size = 1, grad‑accum = 16 → effective batch 16.
* Works without DeepSpeed config; relies only on Accelerate.
* Writes adapter & logs to `/workspace/eval-pipeline/outputs/temporal_lora/`.

Run inside the Docker image:
```bash
docker run --gpus all --rm -it \
  -v $PWD/eval-pipeline:/workspace/eval-pipeline \
  llava-temporal:latest \
  python /workspace/eval-pipeline/run_trl_ft.py
```
"""

from __future__ import annotations

import argparse, os
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset, disable_caching
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer

disable_caching()

# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_id", default="llava-hf/llava-interleave-qwen-7b-hf")
    p.add_argument("--image_root", default="/workspace/eval-pipeline/data")
    p.add_argument("--train_jsonl", default="/workspace/eval-pipeline/data/finetuning_data/llava_temporal_train.jsonl")
    p.add_argument("--eval_jsonl", default="/workspace/eval-pipeline/data/finetuning_data/llava_temporal_dev.jsonl")
    p.add_argument("--output_dir", default="/workspace/eval-pipeline/outputs/temporal_lora")

    # hyper‑params for a single L4 24 GB
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)

    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()

# ------------- Preprocess -------------

def make_preprocess(processor: LlavaProcessor, image_root: Path):
    from PIL import Image

    def _fn(row: Dict):
        imgs = [Image.open(image_root / p) for p in row["images"]]
        prompt = processor.apply_chat_template(row["conversations"], add_generation_prompt=False)
        inputs = processor(text=prompt, images=imgs, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return _fn

# -------------- main ------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading model & processor…")
    processor = LlavaProcessor.from_pretrained(args.model_id)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb_cfg, device_map="auto"
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )

    print("[INFO] Loading dataset…")
    dfiles = {"train": args.train_jsonl}
    if Path(args.eval_jsonl).exists():
        dfiles["eval"] = args.eval_jsonl
    ds = load_dataset("json", data_files=dfiles)
    ds = ds.map(make_preprocess(processor, Path(args.image_root)), remove_columns=ds["train"].column_names, num_proc=8)

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        dataset_text_field=None,
        peft_config=lora_cfg,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        generation_kwargs=dict(max_new_tokens=64),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        output_dir=args.output_dir,
    )

    print("[INFO] Training…")
    trainer.train()

    print("[INFO] Saving adapter & tokenizer…")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("[INFO] Done. Adapter saved at", args.output_dir)


if __name__ == "__main__":
    main()
