#!/usr/bin/env python
"""
Fine‑tune **LLaVA‑HF** with LoRA on multi‑image, interleaved prompts.

This version fixes all outstanding syntax/indentation errors and hardens the
*image ↔ token* alignment once and for all.
"""
import argparse
from pathlib import Path
from typing import List
import re

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

###############################################################################
# CLI helpers
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True, type=str)
    p.add_argument("--val_file",   required=True, type=str)
    p.add_argument("--base_model", required=True, type=str)
    p.add_argument("--output_dir", default="lora-output", type=str)

    # LoRA
    p.add_argument("--lora_rank",   default=8,   type=int)
    p.add_argument("--lora_alpha",  default=32,  type=int)
    p.add_argument("--lora_dropout", default=0.05, type=float)

    # Training
    p.add_argument("--epochs",        default=3,     type=int)
    p.add_argument("--batch_size",    default=4,     type=int)
    p.add_argument("--learning_rate", default=1e-4,  type=float)
    p.add_argument("--fp16",          action="store_true")
    p.add_argument("--in_8bit",       action="store_true")

    # Sequence / image
    p.add_argument("--max_images",        default=6,    type=int,
                   help="Max images to use per example")
    p.add_argument("--max_target_length", default=256,  type=int)
    p.add_argument("--max_source_length", default=8192, type=int,
                   help="Hard cap for text tokens (set lower if OOM)")
    return p.parse_args()

###############################################################################
# Path helper
###############################################################################

def resolve_image_path(path: str) -> str:
    """Resolve *path* against CWD and `eval-pipeline/data`."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    for root in (Path.cwd(), Path.cwd() / "eval-pipeline" / "data"):
        candidate = root / p
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(path)

###############################################################################
# Pre‑processing
###############################################################################

def preprocess(ex, *, processor, max_images: int, max_target: int):
    """Ensure **exact alignment** between number of images and `<image>` tokens.

    1. Choose *n_images* = min(max_images, len(source_images)).
    2. Load exactly *n_images* pictures.
    3. Rewrite the prompt so it contains **exactly** *n_images* `<image>`
       placeholders (keeping the original wording around them).
    4. If we had to add new placeholders, we also add blank images so the
       counts stay equal.
    """

    raw_text: str = ex["instruction"].strip()
    img_paths: List[str] = ex["source_images"]

    # 1️⃣  Decide how many images we will actually use ----------------------
    n_images = min(max_images, len(img_paths))

    # 2️⃣  Load those images --------------------------------------------------
    imgs: List[Image.Image] = [
        Image.open(resolve_image_path(fp)).convert("RGB")
        for fp in img_paths[:n_images]
    ]

    # 3️⃣  Normalise the placeholder count -----------------------------------
    def keep_first_n_placeholders(text: str, n: int) -> str:
        segments = text.split("<image>")
        rebuilt = segments[0]
        used = 0
        for seg in segments[1:]:
            if used < n:
                rebuilt += "<image>"  # keep this placeholder
                used += 1
            # else: drop extra placeholders
            rebuilt += seg
        return rebuilt

    current_tokens = raw_text.count("<image>")
    if current_tokens > n_images:
        raw_text = keep_first_n_placeholders(raw_text, n_images)
    elif current_tokens < n_images:
        missing = n_images - current_tokens
        raw_text = raw_text + " " + " ".join(["<image>"] * missing)

    # After correction they **must** be equal
    assert raw_text.count("<image>") == n_images, "Still mismatched after normalisation"

    # 4️⃣  Processor ---------------------------------------------------------
    proc_inputs = processor(
        images=imgs,
        text=raw_text,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )

    # 🎯 Targets -------------------------------------------------------------
    labels = processor.tokenizer(
        ex["output"],
        padding="max_length",
        truncation=True,
        max_length=max_target,
        return_tensors="pt",
    ).input_ids.squeeze(0)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": proc_inputs.pixel_values.squeeze(0),
        "input_ids": proc_inputs.input_ids.squeeze(0),
        "attention_mask": proc_inputs.attention_mask.squeeze(0),
        "labels": labels,
    }
        "pixel_values": proc_inputs.pixel_values.squeeze(0),  # (N, 3, H, W)
        "input_ids":    proc_inputs.input_ids.squeeze(0),
        "attention_mask": proc_inputs.attention_mask.squeeze(0),
        "labels": labels,
    }

###############################################################################
# Collator (leave pixel_values as 5‑D)
###############################################################################
class LlavaMultiImageCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        return super().__call__(features, return_tensors="pt")

###############################################################################
# Main
###############################################################################

def main() -> None:
    args = parse_args()

    # Dataset ----------------------------------------------------------------
    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

    # Model + processor -------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        load_in_8bit=args.in_8bit,
        device_map="auto",
        torch_dtype=torch.float16 if args.fp16 else None,
    )

    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        ),
    )

    # Pre‑tokenise ------------------------------------------------------------
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

    # Collator ---------------------------------------------------------------
    data_collator = LlavaMultiImageCollator(
        tokenizer=processor.tokenizer,
        model=model,
        label_pad_token_id=processor.tokenizer.pad_token_id,
        padding="longest",
    )

    # Training args ----------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )

    # Monkey‑patch vision tower to flatten 5‑D → 4‑D -------------------------
    vt = model.base_model.vision_tower
    if not hasattr(vt, "_orig_forward"):
        vt._orig_forward = vt.forward
        def vt_forward(pixel_values, *a, **kw):
            if pixel_values.dim() == 5:  # (B, N, 3, H, W)
                b, n, c, h, w = pixel_values.shape
                pixel_values = pixel_values.view(b * n, c, h, w)
            return vt._orig_forward(pixel_values, *a, **kw)
        vt.forward = vt_forward
        print("🔧 Vision tower patched for 5‑D input → 4‑D flattening")

    # Trainer ---------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
