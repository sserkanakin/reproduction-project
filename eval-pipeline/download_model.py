from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch

model_id = "llava-hf/llava-interleave-qwen-7b-hf"

print(f"Downloading model and processor for {model_id}...")

# We don't need the full quantization config here, just enough to download the model files.
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(model_id)

print("Download complete.")