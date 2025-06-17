# eval-pipeline/test_inference_with_dummy_image.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ← after you’ve done the manual registration or trust_remote_code biz…
MODEL_DIR = "checkpoints/llava_merged_0.5b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval().to("cuda")

# 1) Prepare a dummy “image” batch of size 1:
#    match CLIP ViT-L/14-336px input shape: (B, 3, 336, 336)
dummy_images      = torch.zeros(1, 3, 336, 336, device=model.device, dtype=torch.bfloat16)
dummy_image_sizes = [[336, 336]]

# 2) Tokenize a prompt *including* the <image> placeholder tag
prompt = "<image>\n\nOnce upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 3) Call generate, passing images + image_sizes
outputs = model.generate(
    **inputs,
    images=dummy_images,
    image_sizes=dummy_image_sizes,
    max_new_tokens=50
)

print("\nPrompt:   ", prompt)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
