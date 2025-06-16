#!/usr/bin/env python3
# tools/merge_lora_weights.py

import torch
from transformers import BitsAndBytesConfig
from llava.model.builder import load_pretrained_model
from peft import PeftModel

def merge_and_save(
    base_model_name: str,
    lora_weights_dir: str,
    save_dir: str,
    device: str = "cuda",
):
    # 1) load base LLaVA model in fp16
    _, base_model, _, _ = load_pretrained_model(
        base_model_name,          # e.g. "llava-hf/llava-interleave-qwen-0.5b-hf"
        base_model_name,          # base for tokenizer & vision
        base_model_name.split("/")[-1]
    )
    base_model.to(device).half().eval()

    # 2) wrap with LoRA adapters
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_weights_dir,
        is_trainable=False,
    )
    lora_model.eval()

    # 3) merge & save
    print(f"Merging LoRA adapters from {lora_weights_dir} into base model…")
    merged = lora_model.merge_and_unload()  # returns a model with adapters merged in
    print(f"Saving merged model to {save_dir}…")
    merged.save_pretrained(save_dir)
    print("Done.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Base LLaVA model name or path")
    p.add_argument("--lora", required=True,
                   help="Directory of your LoRA adapter weights")
    p.add_argument("--out",   required=True,
                   help="Where to write the merged checkpoint")
    args = p.parse_args()
    merge_and_save(args.base, args.lora, args.out)
