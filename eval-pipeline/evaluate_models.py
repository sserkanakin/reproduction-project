#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import openai
from tqdm import tqdm

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- 1. CONFIGURATION ---
BASE_MODEL_ID = "llava-hf/llava-interleave-qwen-0.5b-hf"
FINETUNED_MODEL_PATH = "eval-pipeline/lmms-finetune/checkpoints/llava-interleave-qwen-0.5b-merged/"
TEST_DATA_PATH = "eval-pipeline/data/finetune_data/test.json"
OPENAI_JUDGE_MODEL = "gpt-4o"

# --- HELPER FUNCTIONS ---
def load_model_and_processor(model_path):
    print(f"Loading model: {model_path}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda")

    processor_path = BASE_MODEL_ID
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    print("...loading complete.")
    return model, processor

def get_model_prediction(model, processor, image_paths, question):
    try:
        images = [Image.open(p).convert("RGB") for p in image_paths]
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return None

    prompt_chat = [{"role": "user", "content": ("<image>" * len(images)) + "\n" + question}]
    prompt_text = processor.tokenizer.apply_chat_template(
        prompt_chat,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(text=prompt_text, images=images, return_tensors="pt").to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def get_option_from_response(openai_client, model_response, options_text):
    judge_prompt = f"""You are an AI assistant. Your task is to determine which of the following options is best represented by the given text. The text provides an explanation for ordering a sequence of images. Please respond with only the letter of the correct option (A, B, C, or D) and nothing else.

**Options:**
{options_text}

**Text to Evaluate:**
{model_response}

**Your Answer (A, B, C, or D):**"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=5,
            temperature=0.0
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

# --- MAIN EVALUATION LOGIC ---
def main():
    try:
        openai_client = openai.OpenAI()
        openai_client.models.list()  # Test API call
    except openai.AuthenticationError:
        print("ERROR: OpenAI API key is missing or invalid.")
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    base_model, base_processor = load_model_and_processor(BASE_MODEL_ID)
    finetuned_model, finetuned_processor = load_model_and_processor(FINETUNED_MODEL_PATH)

    try:
        with open(TEST_DATA_PATH, 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Test data file not found at {TEST_DATA_PATH}")
        return

    base_model_correct = 0
    finetuned_model_correct = 0
    total_samples = len(test_data)

    for i, item in enumerate(tqdm(test_data, desc="Evaluating models")):
        print(f"\n--- Evaluating Sample {i + 1}/{total_samples} ---")
        image_paths = [os.path.join("eval-pipeline/data", path.lstrip("./")) for path in item["input_image_path"]]
        question = item["question"]
        options = item["options"]
        ground_truth = item["output"]

        base_response = get_model_prediction(base_model, base_processor, image_paths, question)
        if base_response:
            print("Base Model Raw Response:", base_response[:100] + "...")
            base_choice = get_option_from_response(openai_client, base_response, options)
            print(f"Judged Choice for Base Model: {base_choice}")
            if base_choice == ground_truth:
                base_model_correct += 1

        finetuned_response = get_model_prediction(finetuned_model, finetuned_processor, image_paths, question)
        if finetuned_response:
            print("Finetuned Model Raw Response:", finetuned_response[:100] + "...")
            finetuned_choice = get_option_from_response(openai_client, finetuned_response, options)
            print(f"Judged Choice for Finetuned Model: {finetuned_choice}")
            if finetuned_choice == ground_truth:
                finetuned_model_correct += 1

    print("\n" + "=" * 20 + " FINAL RESULTS " + "=" * 20)
    base_accuracy = (base_model_correct / total_samples) * 100 if total_samples > 0 else 0
    finetuned_accuracy = (finetuned_model_correct / total_samples) * 100 if total_samples > 0 else 0
    print(f"Total Samples Evaluated: {total_samples}")
    print(f"\nBase Model Accuracy: {base_accuracy:.2f}% ({base_model_correct}/{total_samples})")
    print(f"Finetuned Model Accuracy: {finetuned_accuracy:.2f}% ({finetuned_model_correct}/{total_samples})")
    print("=" * 55)

if __name__ == "__main__":
    main()