from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load your base model (quantization / bf16 / device_map as needed)
base = AutoModelForCausalLM.from_pretrained(
    "llava-hf/llava-interleave-qwen-0.5b-hf",
    torch_dtype="auto", device_map="auto"
)

# 2. Wrap in PEFT and load your adapters
peft = PeftModel.from_pretrained(base, "checkpoints/temporal_lora_0.5b")

# 3. Merge LoRA into base weights, then free the adapter memory
merged = peft.merge_and_unload()

# 4. Save the merged model
merged.save_pretrained("checkpoints/temporal_lora_0.5b_merged")
