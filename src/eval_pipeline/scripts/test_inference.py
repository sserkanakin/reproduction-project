# src/eval_pipeline/scripts/test_inference.py
import os
import shutil
import tempfile
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1) REMOVE the manual registration for LlavaLlamaForCausalLM
# from transformers.models.auto.configuration_auto import CONFIG_MAPPING
# from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
# from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
#
# CONFIG_MAPPING[LlavaConfig.model_type]                      = LlavaConfig
# MODEL_FOR_CAUSAL_LM_MAPPING[LlavaConfig]                   = LlavaLlamaForCausalLM

# Point to your merged‐LoRA folder:
SRC_DIR = "checkpoints/llava_merged_0.5b"


def main():
    # Copy to a temp dir so we can patch config.json
    tmp = tempfile.mkdtemp(prefix="llava_test_")
    dst = os.path.join(tmp, "model")
    shutil.copytree(SRC_DIR, dst)

    # Remove nested text_config (raw dict) to avoid to_dict() errors
    cfgf = os.path.join(dst, "config.json")
    cfg = json.load(open(cfgf))
    if isinstance(cfg.get("text_config"), dict):
        print("🔧 Stripping nested text_config…")
        # This is also a good place to ensure model_type is correct if needed
        # cfg['model_type'] = 'llava_qwen'
        del cfg["text_config"]
        with open(cfgf, "w") as f:
            json.dump(cfg, f)

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        dst,
        trust_remote_code=True,
    )

    print("Loading model…")
    model = AutoModelForCausalLM.from_pretrained(
        dst,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda")  # .eval() is good practice, .to("cuda") might be redundant with device_map

    print("Model loaded successfully.")

    # Test inference
    print("Running inference test…")

    dummy_images = torch.zeros(1, 3, 336, 336, device=model.device, dtype=torch.bfloat16)
    dummy_image_sizes = [[336, 336]]

    prompt = "<image>\nOnce upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        images=dummy_images,
        image_sizes=dummy_image_sizes,
        max_new_tokens=50
    )

    # Minor Typo Fix: Use 'outputs' which you defined, not 'out'
    print("\n> Prompt:   ", prompt)
    print("> Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()