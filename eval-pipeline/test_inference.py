# eval-pipeline/test_inference_workaround.py
import os
import shutil
import tempfile
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Point this at your merged LoRA checkpoint
SRC_DIR = "checkpoints/llava_merged_0.5b"

def main():
    # 1) Copy into a temp folder so we don’t clobber your real checkpoint
    tmpdir = tempfile.mkdtemp(prefix="llava_test_")
    outdir = os.path.join(tmpdir, "model")
    shutil.copytree(SRC_DIR, outdir)

    # 2) Strip out the nested text_config dict so AutoConfig won’t choke
    cfg_path = os.path.join(outdir, "config.json")
    cfg = json.load(open(cfg_path, "r"))
    if isinstance(cfg.get("text_config"), dict):
        print("🔧 Removing nested text_config to avoid .to_dict() error")
        del cfg["text_config"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

    # 3) Load tokenizer + model with trust_remote_code
    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(outdir, trust_remote_code=True)

    print("Loading model…")
    model = AutoModelForCausalLM.from_pretrained(
        outdir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda")

    # 4) Run a quick text‐only prompt
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print("\nPrompt:   ", prompt)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
