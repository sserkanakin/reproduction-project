# test_inference_hf.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "checkpoints/llava_merged_0.5b"

def main():
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,        # <-- allow custom Llava tokenizer
    )

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,        # <-- allow custom LlavaConfig + LlavaLlamaForCausalLM
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval().to("cuda")

    # Simple text‐only prompt
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)

    print("\nPrompt:   ", prompt)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
