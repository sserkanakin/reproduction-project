# inference.py
"""
A script to run inference with a fine-tuned Idefics2 LoRA adapter.
It loads the base model, attaches the trained adapter, and generates a response
for a given set of images and a prompt from the command line.

VERSION 2 UPDATE:
- Fixes a `KeyError` during adapter loading by manually correcting the state
  dictionary keys. This is a robust workaround for a known issue when loading
  adapters saved with the `modules_to_save` argument.

How to Run:
   python inference.py \
       --adapter_path ./idefics2-8b-temporal-finetune-a100/final_checkpoint \
       --image_paths ./data/finetune_data/Continuous-temporal/temporal_ordering/temporal_ordering_150_0.jpg \
                     ./data/finetune_data/Continuous-temporal/temporal_ordering/temporal_ordering_150_1.jpg \
                     # ... and so on for all image paths
"""

import torch
import argparse
import os
from collections import OrderedDict
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from PIL import Image
import logging
from safetensors.torch import load_file

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def main(args):
    # --- 1. Load the base model and processor ---
    logger.info(f"Loading base model from {args.base_model_id}...")
    processor = AutoProcessor.from_pretrained(args.base_model_id)
    model = Idefics2ForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # --- 2. Corrected Adapter Loading ---
    logger.info(f"Loading and correcting fine-tuned LoRA adapter from {args.adapter_path}...")
    try:
        # Load the adapter configuration
        lora_config = LoraConfig.from_pretrained(args.adapter_path)
        
        # Create a PEFT model from the base model and the loaded config
        model = get_peft_model(model, lora_config)
        
        # Manually load the state dict from the safetensors file
        adapter_weights_path = os.path.join(args.adapter_path, "adapter_model.safetensors")
        adapter_weights = load_file(adapter_weights_path)
        
        # Correct the keys by removing the unexpected prefix
        corrected_weights = OrderedDict()
        for key, value in adapter_weights.items():
            # This is the magic fix: remove the incorrect prefix if it exists
            if key.startswith("base_model.model."):
                new_key = key[len("base_model.model."):]
                corrected_weights[new_key] = value
            else:
                corrected_weights[key] = value

        # Load our corrected state dict. `strict=False` is important as we are only loading adapter weights.
        model.load_state_dict(corrected_weights, strict=False)
        
        logger.info("Adapter loaded successfully with corrected keys.")
    except Exception as e:
        logger.error(f"Failed to load and correct adapter. Error: {e}")
        return

    # --- 3. Prepare the inputs ---
    logger.info(f"Preparing {len(args.image_paths)} images and prompt...")
    try:
        images = [Image.open(path).convert("RGB") for path in args.image_paths]
    except FileNotFoundError as e:
        logger.error(f"Error: Image not found at path: {e}")
        return

    # Create the structured content list for the chat template
    content = [{"type": "text", "text": args.prompt}] + [{"type": "image"}] * len(images)
    messages = [{"role": "user", "content": content}]
    
    # The processor handles templating
    prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=[prompt_str],
        images=[images],
        return_tensors="pt"
    ).to("cuda")

    # --- 4. Generate the response ---
    logger.info("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # --- 5. Print the output ---
    print("\n" + "="*50)
    print("      MODEL OUTPUT")
    print("="*50 + "\n")
    # Clean up the output to remove the prompt part
    if "Assistant:" in generated_texts[0]:
        cleaned_output = generated_texts[0].split("Assistant:")[1].strip()
        print(cleaned_output)
    else:
        print(generated_texts[0]) # Print raw output if template isn't found
    print("\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Idefics2 model.")
    parser.add_argument("--base_model_id", type=str, default="HuggingFaceM4/idefics2-8b")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the saved LoRA adapter checkpoint directory.")
    parser.add_argument("--image_paths", nargs="+", required=True, help="List of paths to the input images.")
    parser.add_argument("--prompt", type=str, default="\nPlease predict the order of the following pictures, and give each picture a sequential index. This index starts from 0. The larger the index, the later the order.", help="The text prompt to accompany the images.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()
    main(args)
