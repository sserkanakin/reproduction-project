#!/usr/bin/env python
"""run_trl_ft.py ── Flexible QLoRA fine‑tune for LLaVA‑Interleave‑Qwen‑7B.

* Works on **GPU (CUDA)** with 4‑bit NF4 quantisation.
* Works on **CPU/MPS** for logic smoke‑tests by loading a *tiny* stub model
  or the full weights if you have enough RAM.

Quick M‑series Mac test (tiny model):
```bash
python run_trl_ft.py \
  --model_id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --train_jsonl ./data/finetuning_data/test/llava_temporal_train.jsonl \
  --output_dir ./outputs/test_cpu \
  --cpu --quant none --num_train_epochs 0.01 --max_seq_length 512
```
Docker GPU full run stays unchanged.
"""
from __future__ import annotations

import argparse, os
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset, disable_caching
from transformers import (
    AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

disable_caching()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Runtime / quant
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA visible")
    p.add_argument("--quant", choices=["4bit", "none"], default="4bit")

    # Paths
    p.add_argument("--model_id", default="llava-hf/llava-interleave-qwen-7b-hf")
    p.add_argument("--image_root", default="./data")
    p.add_argument("--train_jsonl", default="./data/finetuning_data/llava_temporal_train.jsonl")
    p.add_argument("--eval_jsonl", default="./data/finetuning_data/llava_temporal_dev.jsonl")
    p.add_argument("--output_dir", default="./outputs/temporal_lora")

    # Training params
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)

    # Model / LoRA params
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=500)

    return p.parse_args()

# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------

def make_preprocess(proc: AutoProcessor, img_root: Path):
    from PIL import Image

    def _fn(row: Dict):
        imgs = [Image.open(img_root / p) for p in row.get("images", [])]
        prompt = proc.apply_chat_template(row["conversations"], add_generation_prompt=False)
        out = proc(text=prompt, images=imgs, return_tensors="pt")
        out["labels"] = out["input_ids"].clone()
        return out

    return _fn

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # ---------------- Choose quant mode -------------------
    gpu_available = torch.cuda.is_available() and not args.cpu and args.quant == "4bit"

    if gpu_available:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        quant_kwargs = dict(quantization_config=bnb_cfg, device_map="auto")
    else:
        print("[INFO] Falling back to full‑precision on CPU/MPS – expect slow run & high RAM.")
        quant_kwargs = dict(device_map={"": "cpu"})

    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **quant_kwargs)

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM", bias="none")

    files = {"train": args.train_jsonl}
    if Path(args.eval_jsonl).exists():
        files["eval"] = args.eval_jsonl
    ds = load_dataset("json", data_files=files)
    ds = ds.map(make_preprocess(processor, Path(args.image_root)), remove_columns=ds["train"].column_names, num_proc=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=gpu_available,
        report_to=["tensorboard"],
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        dataset_text_field=None,
        peft_config=lora_cfg,
        max_seq_length=args.max_seq_length,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
