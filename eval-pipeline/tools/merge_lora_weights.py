#!/usr/bin/env python3
# eval-pipeline/tools/merge_lora.py

import argparse
import torch
from pathlib import Path

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from peft import PeftModel

def merge_lora(base_model_id: str, lora_dir: str, save_dir: str):
    # disable any random init side-effects
    disable_torch_init()

    # 1) load base LLaVA model + tokenizer + image processor
    #    model_name is just the folder name of your LoRA checkpoint
    model_name = Path(lora_dir).name
    tokenizer, model, image_processor, _ = load_pretrained_model(
        base_model_id,         # e.g. "llava-hf/llava-interleave-qwen-0.5b-hf"
        base_model_id,         # same here
        model_name             # this folder holds your adapters
    )

    # move to CPU so we can merge without eating GPU memory
    model.cpu()

    # 2) wrap in PEFT and merge adapters
    peft_model = PeftModel.from_pretrained(
        model,
        lora_dir,
        is_trainable=False
    )
    merged: torch.nn.Module = peft_model.merge_and_unload()

    # 3) save the result (tokenizer + merged model)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"✅ merged LoRA weights saved to {save_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Merge LoRA adapters back into the base LLaVA model"
    )
    p.add_argument("--base_model",  type=str, required=True,
                   help="HuggingFace ID or path of original LLaVA base")
    p.add_argument("--lora_dir",    type=str, required=True,
                   help="Directory containing your LoRA adapter files")
    p.add_argument("--save_dir",    type=str, required=True,
                   help="Where to write out the merged checkpoint")
    args = p.parse_args()
    merge_lora(args.base_model, args.lora_dir, args.save_dir)
