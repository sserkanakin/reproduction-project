from llava.model.builder import load_pretrained_model
import torch

# 1. Load your merged LoRA model onto the GPU
base_model = "llava-hf/llava-interleave-qwen-0.5b-hf"
model_dir  = "checkpoints/temporal_merged_0.5b"
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_dir,
    base_model,
    None,
    device_map="auto"
)
model = model.to("cuda").eval()

# 2. Run a simple text‐only prompt
prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)

# 3. Decode and print
print("Prompt:   ", prompt)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
