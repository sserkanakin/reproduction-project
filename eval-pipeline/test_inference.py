# test_inference_hf.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "checkpoints/temporal_merged_0.5b"

def main():
    # 1) Load tokenizer & model with trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,     # or torch.float16 if you prefer
                ).eval()

    # 2) Simple text‐only prompt
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)

    # 3) Decode & print
    print("Prompt:   ", prompt)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
