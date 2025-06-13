#!/usr/bin/env python
"""run_trl_ft.py ── Flexible fine‑tune helper

* **GPU (CUDA)** → 4‑bit NF4 QLoRA for full *LLaVA‑Interleave‑Qwen‑7B*.
* **CPU/MPS** → loads either a *tiny text‑only stub* **or** any single‑image
  LLaVA variant for logic tests (no bitsandbytes required).

Usage examples
--------------
### 1. Mac smoke‑test (text‑only stub, no vision)
```bash
python run_trl_ft.py \
  --model_id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --train_jsonl ./data/finetuning_data/test/llava_temporal_train.jsonl \
  --output_dir ./outputs/test_stub \
  --cpu --quant none --num_train_epochs 0.01 --max_seq_length 512
```
### 2. L4 GPU full fine‑tune (Docker)
```bash
docker run --gpus all --rm -it \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
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
from transformers import (
    AutoProcessor, AutoConfig, AutoModelForCausalLM,
    LlavaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

disable_caching()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA visible")
    ap.add_argument("--quant", choices=["4bit", "none"], default="4bit",
                    help="4bit only valid on CUDA")

    ap.add_argument("--model_id", default="llava-hf/llava-interleave-qwen-7b-hf")
    ap.add_argument("--image_root", default="./data")
    ap.add_argument("--train_jsonl", default="./data/finetuning_data/llava_temporal_train.jsonl")
    ap.add_argument("--eval_jsonl", default="./data/finetuning_data/llava_temporal_dev.jsonl")
    ap.add_argument("--output_dir", default="./outputs/temporal_lora")

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=1e-4)

    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=500)
    return ap.parse_args()

# ---------------------------------------------------------------------------
# Preprocess builder
# ---------------------------------------------------------------------------

def make_preprocess(proc: AutoProcessor, img_root: Path, expects_images: bool):
    """Returns a row‑level mapper that:
    * builds a text prompt from the last user turn when the model is *not* LLaVA
    * keeps the original chat template + images when LLaVA
    """
    from PIL import Image

    def _fn(row: Dict):
        if expects_images:
            prompt = proc.apply_chat_template(row["conversations"], add_generation_prompt=False)
            imgs = [Image.open(img_root / p) for p in row["images"]]
            inputs = proc(text=prompt, images=imgs, return_tensors="pt")
        else:
            # text‑only stub → just feed the last human utterance
            prompt = row["conversations"][-1]["content"]
            inputs = proc(prompt, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return _fn

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # determine model family
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    is_llava = cfg.model_type == "llava"

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # choose quant
    use_cuda_4bit = torch.cuda.is_available() and not args.cpu and args.quant == "4bit" and is_llava
    if use_cuda_4bit:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        quant_kwargs = dict(quantization_config=bnb_cfg, device_map="auto")
    else:
        print("[INFO] CPU/MPS or text‑only model → full precision.")
        quant_kwargs = dict(device_map={"": "cpu"})

    # load model
    if is_llava:
        model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **quant_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **quant_kwargs)

    # LoRA config (only attaches to linear sub‑modules; safe for both model types)
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules="all-linear", task_type="CAUSAL_LM", bias="none")

    # dataset
    files = {"train": args.train_jsonl}
    if Path(args.eval_jsonl).exists():
        files["eval"] = args.eval_jsonl
    ds = load_dataset("json", data_files=files)

    preprocess_fn = make_preprocess(processor, Path(args.image_root), expects_images=is_llava)
    ds = ds.map(preprocess_fn, remove_columns=ds["train"].column_names, num_proc=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=use_cuda_4bit,
        report_to=[],  # disable tensorboard for ultra‑light envs; add back on GPU
        gradient_checkpointing=use_cuda_4bit,
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
