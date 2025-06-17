#!/usr/bin/env python3
# eval-pipeline/tools/merge_lora.py

import argparse
import torch
from pathlib import Path

# 1) LLaVA’s own loader
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model

# 2) PEFT wrapper
from peft import PeftModel

def merge_lora(base_model_id: str, lora_dir: str, save_dir: str):
    disable_torch_init()

    # load LLaVA model + tokenizer + image processor
    model_name = Path(lora_dir).name
    tokenizer, model, _, _ = load_pretrained_model(
        base_model_id,    # e.g. "llava-hf/llava-interleave-qwen-0.5b-hf"
        base_model_id,    # same as above
        model_name        # folder where your LoRA adapters live
    )
    print("✅ Base model and tokenizer loaded from", base_model_id)
    # move to CPU before merging
    model.cpu()

    # wrap in PEFT and merge
    peft = PeftModel.from_pretrained(model, lora_dir, is_trainable=False)
    merged_model: torch.nn.Module = peft.merge_and_unload()

    # save merged model + tokenizer
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"✅ LoRA adapters merged into base and saved to {save_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True,
                   help="HuggingFace ID or path of the original LLaVA base")
    p.add_argument("--lora_dir",   type=str, required=True,
                   help="Directory containing your LoRA adapter files")
    p.add_argument("--save_dir",   type=str, required=True,
                   help="Where to write the merged checkpoint")
    args = p.parse_args()
    merge_lora(args.base_model, args.lora_dir, args.save_dir)
