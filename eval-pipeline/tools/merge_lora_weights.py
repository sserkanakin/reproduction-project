from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import llava.model.language_model.llava_llama

BASE = "llava-hf/llava-interleave-qwen-0.5b-hf"
LORA = "checkpoints/temporal_lora_0.5b"
OUT  = "checkpoints/temporal_lora_0.5b_merged"

# 1) Load your base model (4-bit + BF16 compute)
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=False)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bf16",
)

# 2) Overlay the LoRA adapters
model = PeftModel.from_pretrained(base, LORA)

# 3) Merge adapters into the base weights & unload PEFT wrappers
merged = model.merge_and_unload()

# 4) Save the merged, standalone checkpoint
merged.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)

print(f"Merged checkpoint written to {OUT}")
