#!/usr/bin/env python
"""
Fine‑tune **LLaVA‑HF** with LoRA on a vision‑language ordering task.

Main differences to the first draft
-----------------------------------
* **No truncation** of the text sequence ⇒ all `<image>` tokens are preserved, so
  the “Mismatch in image token count” error disappears.
* `max_source_length` bumped to **8192** (CLI flag) – adjust if you have plenty
  of GPU RAM, otherwise lower it.
* Collator still collapses `(B, N, 3, H, W)` → `(B·N, 3, H, W)` for SigLIP.
* Padding moved entirely to the collator (`padding="longest"`).
* Added gradient checkpointing & FP16 defaults that work well on 24 GB GPUs.

Run example
-----------
```bash
python finetune_llava_lora.py \
  --train_file data/train.jsonl \
  --val_file   data/val.jsonl   \
  --base_model llava-hf/llava-interleave-qwen-7b-hf \
  --output_dir output/lora-llava \
  --batch_size 2 --epochs 1 --fp16
```
"""
import argparse
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--val_file", type=str, required=True)
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="lora-output")

    # LoRA params
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training hyper‑params
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--in_8bit", action="store_true")

    # Sequence/image handling
    p.add_argument("--max_images", type=int, default=6,
                   help="Images per example (extra images are dropped, fewer are padded)")
    p.add_argument("--max_target_length", type=int, default=256)
    p.add_argument("--max_source_length", type=int, default=8192,
                   help="Hard cap for the tokenised instruction (keep all <image> tokens)")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def resolve_image_path(path: str) -> str:
    """Return an absolute path for *path* (tries CWD and eval‑pipeline/data)."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    for root in (Path.cwd(), Path.cwd() / "eval-pipeline" / "data"):
        candidate = root / p
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(path)

# -----------------------------------------------------------------------------
# Dataset preprocessing
# -----------------------------------------------------------------------------

def preprocess(ex, *, processor, max_images: int, max_target: int):
    # 1️⃣  Images --------------------------------------------------------------
    imgs: List[Image.Image] = []
    for fp in ex["source_images"][:max_images]:
        imgs.append(Image.open(resolve_image_path(fp)).convert("RGB"))
    # pad with black squares if needed
    while len(imgs) < max_images:
        imgs.append(Image.new("RGB", imgs[0].size, (0, 0, 0)))

    # 2️⃣  Processor (no truncation ⇒ all <image> tokens kept) ----------------
    proc_inputs = processor(
        images=imgs,
        text=ex["instruction"],
        padding=False,          # pad later in the collator
        truncation=False,
        return_tensors="pt",
    )

    # 3️⃣  Targets ------------------------------------------------------------
    labels = processor.tokenizer(
        ex["output"],
        padding="max_length",
        truncation=True,
        max_length=max_target,
        return_tensors="pt",
    ).input_ids.squeeze(0)
    labels[labels == processor.tokenizer.pad_token_id] = -100  # ignore padding in loss

    return {
        "pixel_values": proc_inputs.pixel_values.squeeze(0),  # (N, 3, H, W)
        "input_ids": proc_inputs.input_ids.squeeze(0),
        "attention_mask": proc_inputs.attention_mask.squeeze(0),
        "labels": labels,
    }

# -----------------------------------------------------------------------------
# Collator: collapse image dimension
# -----------------------------------------------------------------------------
class LlavaMultiImageCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors="pt")
        pix = batch["pixel_values"]  # (B, N, 3, H, W)
        if pix.ndim == 5:
            b, n, c, h, w = pix.shape
            batch["pixel_values"] = pix.view(b * n, c, h, w)
            batch["num_images"] = torch.tensor([n] * b)
        return batch

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    data_files = {"train": args.train_file, "validation": args.val_file}

    print("📚 Loading dataset …")
    ds = load_dataset("json", data_files=data_files)

    print(f"🧊 Loading base model {args.base_model} …")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        load_in_8bit=args.in_8bit,
        device_map="auto",
        torch_dtype=torch.float16 if args.fp16 else None,
    )

    # LoRA adapter
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        ),
    )

    # Pre‑processing map ------------------------------------------------------
    print("🔄 Tokenising & processing images …")
    ds = ds.map(
        lambda ex: preprocess(
            ex,
            processor=processor,
            max_images=args.max_images,
            max_target=args.max_target_length,
        ),
        remove_columns=list(ds["train"].column_names),
    )
    ds.set_format("torch", ["pixel_values", "input_ids", "attention_mask", "labels"])

    # Collator / Trainer ------------------------------------------------------
    data_collator = LlavaMultiImageCollator(
        tokenizer=processor.tokenizer,
        model=model,
        label_pad_token_id=processor.tokenizer.pad_token_id,
        padding="longest",
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,

        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )

    print("🚀 Starting training …")
    trainer.train()

    print("💾 Saving LoRA adapters →", args.output_dir)
    model.save_pretrained(args.output_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()
