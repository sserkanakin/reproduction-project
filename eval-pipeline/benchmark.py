# benchmark.py
"""
A comprehensive script to benchmark a fine-tuned Idefics2 model against its base version.

VERSION 2.0:
- Handles complex, multiple-choice questions from a JSON file.
- Uses an OpenAI model for robust, intelligent parsing of model outputs.
- Logs detailed raw results and parsed answers to a JSON file.
- Generates a clean text file with a final accuracy summary.
- Includes all previous fixes for model loading and data handling.

How to Run:
   # First, ensure you have a .env file with your OPENAI_API_KEY
   python benchmark.py \
       --val_file data/finetune_data/test.json \
       --image_base_path data/ \
       --adapter_path idefics2-8b-temporal-finetune-a100/final_checkpoint \
       --results_json_path results.json \
       --summary_txt_path summary.txt
"""

import torch
import argparse
import os
import json
import re
from collections import OrderedDict
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import logging
from safetensors.torch import load_file
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---
# Load environment variables from .env file
load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# --- OpenAI-Powered Parsing ---
def parse_with_openai(client: OpenAI, model_output: str, options: str) -> str | None:
    """Uses an OpenAI model to parse the chosen option from the model's text output."""
    try:
        system_prompt = """You are an expert JSON parsing agent. A user will provide you with a text block from a language model and a list of options (A, B, C, D). Your sole job is to identify which single option the model chose. Respond ONLY with a JSON object containing a single key, "chosen_option", with the corresponding letter as the value. For example: {"chosen_option": "A"}. If you cannot determine the choice, return "N/A"."""
        
        user_prompt = f"""Model Output:
---
{model_output}
---

Options:
---
{options}
---
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("chosen_option", "N/A")
    except Exception as e:
        logger.error(f"OpenAI parsing failed: {e}")
        return None

# --- Model Inference ---
def get_model_prediction(model, processor, images, prompt_text, max_new_tokens):
    """Generates a response from a given model."""
    content = [{"type": "text", "text": prompt_text}] + [{"type": "image"}] * len(images)
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(text=[prompt], images=[images], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=processor.tokenizer.eos_token_id)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    assistant_response = generated_texts[0].split("Assistant:")[-1].strip()
    return assistant_response

# --- Main Evaluation Loop ---
def evaluate_model(model, oai_client, processor, dataset, image_base_path, max_new_tokens, model_name="Model"):
    """Evaluates a model and returns accuracy and detailed results."""
    correct_predictions = 0
    total_samples = 0
    results_log = []
    
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataset, desc=f"Evaluating {model_name}"):
            total_samples += 1
            
            try:
                image_paths = [os.path.join(image_base_path, path.lstrip('./')) for path in sample["input_image_path"]]
                images = [Image.open(path).convert("RGB") for path in image_paths]
            except (FileNotFoundError, TypeError) as e:
                logger.warning(f"Skipping sample due to invalid image data: {e}")
                total_samples -= 1
                continue

            # Construct the full prompt
            prompt_text = f"{sample['question']}\n{sample['context']}\n{sample['options']}"
            ground_truth_option = sample["output"]
            
            # Get model prediction
            raw_output = get_model_prediction(model, processor, images, prompt_text, max_new_tokens)
            
            # Parse with OpenAI
            parsed_option = parse_with_openai(oai_client, raw_output, sample['options'])

            # Compare and score
            is_correct = (parsed_option == ground_truth_option)
            if is_correct:
                correct_predictions += 1
            
            # Log detailed results
            results_log.append({
                "sample_source": sample.get("source", "N/A"),
                "prompt": prompt_text,
                "raw_output": raw_output,
                "parsed_option": parsed_option,
                "ground_truth": ground_truth_option,
                "is_correct": is_correct
            })
                
    if total_samples == 0:
        logger.error("No valid samples were found in the evaluation dataset.")
        return 0.0, []
        
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy, results_log

# --- Utility Functions ---
def load_tuned_model(base_model_id, adapter_path):
    """Loads the base model and applies the fine-tuned LoRA adapter with key correction."""
    logger.info("Loading fine-tuned model...")
    model = Idefics2ForConditionalGeneration.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map="auto")
    lora_config = LoraConfig.from_pretrained(adapter_path)
    model = get_peft_model(model, lora_config)
    
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found at {adapter_weights_path}")
        
    adapter_weights = load_file(adapter_weights_path)
    corrected_weights = OrderedDict()
    for key, value in adapter_weights.items():
        if key.startswith("base_model.model."):
            # This logic might need adjustment if keys are still incorrect
            new_key = key[len("base_model.model."):]
            corrected_weights[new_key] = value
        else:
            corrected_weights[key] = value
            
    model.load_state_dict(corrected_weights, strict=False)
    logger.info("Fine-tuned model loaded successfully.")
    return model

def save_results(json_path, summary_path, base_results, tuned_results, base_accuracy, tuned_accuracy):
    """Saves the detailed JSON log and the final summary text file."""
    # Save detailed JSON log
    full_log = {
        "base_model_results": base_results,
        "tuned_model_results": tuned_results
    }
    with open(json_path, 'w') as f:
        json.dump(full_log, f, indent=4)
    logger.info(f"Detailed results saved to {json_path}")

    # Save summary text file
    improvement = tuned_accuracy - base_accuracy
    summary_content = f"""
==================================
      BENCHMARK SUMMARY
==================================
Base Model Accuracy:      {base_accuracy:.2f}%
Fine-Tuned Model Accuracy: {tuned_accuracy:.2f}%
----------------------------------
Accuracy Improvement:     {improvement:+.2f}%
==================================
"""
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    logger.info(f"Summary report saved to {summary_path}")
    print(summary_content)


def main(args):
    # --- 1. Initialize Clients and Load Data ---
    logger.info("Initializing OpenAI client...")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not openai_client.api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    logger.info(f"Loading processor from {args.base_model_id}...")
    processor = AutoProcessor.from_pretrained(args.base_model_id)
    
    logger.info(f"Loading validation dataset from {args.val_file}...")
    val_dataset = load_dataset("json", data_files=args.val_file, split="train")

    # --- 2. Evaluate Base Model ---
    logger.info("Loading BASE model for evaluation...")
    base_model = Idefics2ForConditionalGeneration.from_pretrained(
        args.base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_accuracy, base_results = evaluate_model(
        base_model, openai_client, processor, val_dataset, args.image_base_path, args.max_new_tokens, "Base Model"
    )
    del base_model
    torch.cuda.empty_cache()

    # --- 3. Evaluate Fine-Tuned Model ---
    tuned_model = load_tuned_model(args.base_model_id, args.adapter_path)
    tuned_accuracy, tuned_results = evaluate_model(
        tuned_model, openai_client, processor, val_dataset, args.image_base_path, args.max_new_tokens, "Fine-Tuned Model"
    )

    # --- 4. Save All Results ---
    save_results(args.results_json_path, args.summary_txt_path, base_results, tuned_results, base_accuracy, tuned_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a fine-tuned Idefics2 model against its base version.")
    parser.add_argument("--base_model_id", type=str, default="HuggingFaceM4/idefics2-8b")
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--image_base_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--results_json_path", type=str, default="results.json")
    parser.add_argument("--summary_txt_path", type=str, default="summary.txt")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)
