# test_inference_hf.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ← point here at your actual merged folder
MODEL_DIR = "checkpoints/llava_merged_0.5b"

def main():
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("Loading model")
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                ).eval()
    print("Model loaded successfully")
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Inference will be slow.")

    print("Loading data")
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("Data loaded successfully")
    print("Generating output")
    outputs = model.generate(**inputs, max_new_tokens=50)

    print("Prompt:   ", prompt)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
