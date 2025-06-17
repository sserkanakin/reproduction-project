from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "checkpoints/llava_merged_0.5b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval().to("cuda")

prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out    = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
