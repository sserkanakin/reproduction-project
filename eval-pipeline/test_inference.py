# eval-pipeline/test_inference_builder.py
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import preprocess_images

# 1) Paths
# ──────────────────────────────────────────────────────────────────────────────
# This is your merged LoRA checkpoint:
MODEL_PATH = "checkpoints/llava_merged_0.5b"
# And the base model ID you originally fine-tuned from:
MODEL_BASE = "llava-hf/llava-interleave-qwen-0.5b-hf"

# 2) Load everything on GPU
# ──────────────────────────────────────────────────────────────────────────────
print("Loading checkpoint via builder.load_pretrained_model…")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    checkpoint_path=MODEL_PATH,
    model_base=MODEL_BASE,
    device="cuda",
)

model.eval()

# 3) Prepare dummy image batch
# ──────────────────────────────────────────────────────────────────────────────
# image_processor expects a list of PIL Images or tensors, but we can
# create a zero‐tensor in CLIP’s expected shape (3×336×336) directly:
dummy_image = torch.zeros(3, 336, 336, dtype=torch.float32)  # CLIP expects float32
pixel_values = image_processor(dummy_image, return_tensors="pt")["pixel_values"]
# Batch it:
pixel_values = pixel_values.to("cuda")  # shape (1,3,336,336)
image_sizes  = [[336, 336]]

# 4) Tokenize a prompt (with the <image> placeholder)
# ──────────────────────────────────────────────────────────────────────────────
prompt = "<image>\nOnce upon a time"
inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to("cuda")

# 5) Generate
# ──────────────────────────────────────────────────────────────────────────────
print("Running generate()…")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        images=pixel_values,
        image_sizes=image_sizes,
        max_new_tokens=50,
    )

# 6) Decode
# ──────────────────────────────────────────────────────────────────────────────
generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("\n> Prompt:   ", prompt)
print("> Generated:", generated)
