# eval-pipeline/test_inference.py
import torch

# 1) Register LLaVA config + model in the HF registry:
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

import llava.model.language_model.llava_llama as llava_llama_mod

# Map the string in config.json → the Python class
CONFIG_MAPPING["llava_llama"] = llava_llama_mod.LlavaConfig
# Map the config class → the AutoModel class
MODEL_FOR_CAUSAL_LM_MAPPING[llava_llama_mod.LlavaConfig] = llava_llama_mod.LlavaLlamaForCausalLM

# 2) Now import the standard HF Auto classes
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "checkpoints/llava_merged_0.5b"

def main():
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,  # still required for any custom tokenizers
    )

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,  # picks up custom generate() etc.
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda")

    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)

    print("\nPrompt:   ", prompt)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
