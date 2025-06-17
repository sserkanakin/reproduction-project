# eval-pipeline/test_inference_final.py
import os
import shutil
import tempfile
import json
import torch

# 1) Manually register LlavaConfig → LlavaLlamaForCausalLM in HF’s Auto mappings
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

CONFIG_MAPPING[LlavaConfig.model_type]                      = LlavaConfig
MODEL_FOR_CAUSAL_LM_MAPPING[LlavaConfig]                   = LlavaLlamaForCausalLM

# 2) Now import the HF factories
from transformers import AutoTokenizer, AutoModelForCausalLM

# Point to your merged‐LoRA folder:
SRC_DIR = "checkpoints/llava_merged_0.5b"

def main():
    # Copy to a temp dir so we can patch config.json
    tmp = tempfile.mkdtemp(prefix="llava_test_")
    dst = os.path.join(tmp, "model")
    shutil.copytree(SRC_DIR, dst)

    # Remove nested text_config (raw dict) to avoid to_dict() errors
    cfgf = os.path.join(dst, "config.json")
    cfg  = json.load(open(cfgf))
    if isinstance(cfg.get("text_config"), dict):
        print("🔧 Stripping nested text_config…")
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
    ).eval().to("cuda")

    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out    = model.generate(**inputs, max_new_tokens=50)
    print("\n> Prompt:   ", prompt)
    print("> Generated:", tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
