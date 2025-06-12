#!/usr/bin/env python
"""
Fine‑tune **LLaVA‑HF** with LoRA on a vision‑language ordering task.

**Patch 3 – HF compatibility**
--------------------------------
Some older `transformers` builds (≈ < 4.0) do **not** expose the
`evaluation_strategy` argument in `TrainingArguments`.  This revision:

* Drops `evaluation_strategy` and `eval_steps` – evaluation is now run
  **once at the end** via an explicit `trainer.evaluate()` call.
* Keeps the rest of the training loop identical.

If you upgrade to a more recent `transformers` (≥ 4.0) you can bring back
step‑wise evaluation by re‑adding the two lines that were removed.
"""
import argparse
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

###############################################################################
# CLI
###############################################################################

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
                   help="Keep all <image> tokens; lower if you OOM")
    return p.parse_args()

###############################################################################
# Helpers
###############################################################################

def resolve_image_path(path: str) -> str:
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
    imgs: List[Image.Image] = [
        Image.open(resolve_image_path(fp)).convert("RGB")
        for fp in ex["source_images"][:max_images]
    ]
    while len(imgs) < max_images:
        imgs.append(Image.new("RGB", imgs[0].size, (0, 0, 0)))

    proc_inputs = processor(
        images=imgs,
        text=ex["instruction"],
        padding=False,
        truncation=False,
        return_tensors="pt",
    )

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

###############################################################################
# Collator
###############################################################################
class LlavaMultiImageCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors="pt")
        pix = batch["pixel_values"]
        if pix.ndim == 5:  # (B, N, 3, H, W) → (B·N, 3, H, W)
            b, n, c, h, w = pix.shape
            batch["pixel_values"] = pix.view(b * n, c, h, w)
            batch["num_images"] = torch.tensor([n] * b)
        return batch

###############################################################################
# Main
###############################################################################

def main():
    args = parse_args()
    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

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

    ds = ds.map(
        lambda ex: preprocess(ex, processor=processor, max_images=args.max_images, max_target=args.max_target_length),
        remove_columns=list(ds["train"].column_names),
    )
    ds.set_format("torch", ["pixel_values", "input_ids", "attention_mask", "labels"])

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

    trainer.train()

    # One evaluation pass at the end (works on any HF version)
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
