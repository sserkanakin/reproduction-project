# inference_with_manual_registration.py
import torch

# 1) Bring in the HF registries
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

# 2) Import LLaVA’s config+model classes
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

# 3) Register them
CONFIG_MAPPING[LlavaConfig.model_type]                   = LlavaConfig
MODEL_FOR_CAUSAL_LM_MAPPING[LlavaConfig]                = LlavaLlamaForCausalLM

# 4) Now import the Auto classes
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "checkpoints/llava_merged_0.5b"

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,    # still needed for custom code bits
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda")

    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
