#!/usr/bin/env python
"""run_trl_ft.py — one‑GPU 4‑bit QLoRA fine‑tune for **LLaVA‑Interleave‑Qwen‑7B**

No external `llava` package required.  A self‑contained `collate_fn` does:
1. **Flatten** 5‑D `pixel_values` → `(B·N, 3, H, W)`.
2. Pad `input_ids`, `labels`, `attention_mask` with zeros.

This avoids all import/install headaches and works on a single NVIDIA L4
(24 GB) with batch 1 × grad‑accum 16.
"""
from __future__ import annotations

import argparse, os
from pathlib import Path
from typing import Dict, List

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
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="llava-hf/llava-interleave-qwen-7b-hf")
    p.add_argument("--image_root", default="/workspace/eval-pipeline/data")
    p.add_argument("--train_jsonl", default="/workspace/eval-pipeline/data/finetuning_data/llava_temporal_train.jsonl")
    p.add_argument("--eval_jsonl", default="/workspace/eval-pipeline/data/finetuning_data/llava_temporal_dev.jsonl")
    p.add_argument("--output_dir", default="/workspace/eval-pipeline/outputs/temporal_lora")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()

# ---------------------------------------------------------------------------
# Pre‑process — leave pixel tensor 5‑D; collator will flatten
# ---------------------------------------------------------------------------

def make_preprocess(proc: AutoProcessor, img_root: Path):
    from PIL import Image

    def _fn(row: Dict):
        prompt = proc.apply_chat_template(row["conversations"], add_generation_prompt=False)
        imgs = [Image.open(img_root / p) for p in row["images"]]
        inputs = proc(text=prompt, images=imgs, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return _fn

# ---------------------------------------------------------------------------
# Collator — flatten images & pad token tensors
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]):
    """Build a batch for LLaVA.
    * Flattens images to 4‑D `(B·N,3,H,W)` required by SigLIP.
    * Pads token fields with zeros (pad‑id 0).
    """
    # ── images ─────────────────────────────────────────────────────────────
    img_tensors = []
    for ex in batch:
        pv = ex["pixel_values"]  # list[N] or tensor
        if isinstance(pv, list):
            pv = torch.stack([torch.as_tensor(t) for t in pv], dim=0)  # (N,3,H,W)
        if pv.ndim == 5:                                              # (1,N,3,H,W)
            pv = pv.squeeze(0)
        pv = pv.view(-1, *pv.shape[-3:])                              # (N,3,H,W)
        img_tensors.append(pv)
    pixel_values = torch.cat(img_tensors, dim=0)                      # (B·N,3,H,W)

    # ── tokens ─────────────────────────────────────────────────────────────
    def _pad(name: str):
        seqs = [torch.as_tensor(ex[name], dtype=torch.long) for ex in batch]
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)

    return {
        "pixel_values":   pixel_values,
        "input_ids":      _pad("input_ids"),
        "labels":         _pad("labels"),
        "attention_mask": _pad("attention_mask"),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 4‑bit NF4 QLoRA ------------------------------------------
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb_cfg, device_map="auto"
    )

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules="all-linear", task_type="CAUSAL_LM", bias="none")

    # dataset ---------------------------------------------------
    files = {"train": args.train_jsonl}
    if Path(args.eval_jsonl).exists():
        files["eval"] = args.eval_jsonl
    ds = load_dataset("json", data_files=files)
    ds = ds.map(make_preprocess(processor, Path(args.image_root)), remove_columns=ds["train"].column_names, num_proc=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        report_to=["tensorboard"],
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        data_collator=collate_fn,
        dataset_text_field=None,
        peft_config=lora_cfg,
        max_seq_length=args.max_seq_length,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
