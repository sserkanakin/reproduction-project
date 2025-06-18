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
# BASE_MODEL_ID = "llava-hf/llava-interleave-qwen-0.5b-hf"
BASE_MODEL_ID = "llava-hf/llava-interleave-qwen-7b-hf"  # Base model ID
# FINETUNED_MODEL_PATH = "eval-pipeline/lmms-finetune/checkpoints/llava-interleave-qwen-0.5b-merged/"
FINETUNED_MODEL_PATH = "eval-pipeline/lmms-finetune/checkpoints/llava-interleave-qwen-7b-merged/"  # Finetuned model path
TEST_DATA_PATH = "eval-pipeline/data/finetune_data/test.json"
OPENAI_JUDGE_MODEL = "gpt-4o"

# --- HELPER FUNCTIONS ---
def load_model_and_processor(model_path):
    print(f"Loading model: {model_path}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    ).to("cuda")

    processor_path = BASE_MODEL_ID
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    print("...loading complete.")
    return model, processor


def get_model_prediction(model, processor, image_paths, question):
    """
    Generates a free-text response from a model using the modern `apply_chat_template` method.
    This version has the CORRECTED dictionary format for images.
    """
    # 1. Add a debug print to verify the absolute path of the first image
    if image_paths:
        first_image_abs_path = os.path.abspath(image_paths[0])
        print(f"DEBUG: Attempting to load first image from absolute path: {first_image_abs_path}")
        if not os.path.exists(first_image_abs_path):
            print(f"WARNING: This image path does not exist!")

    # 2. Construct the 'content' list for the chat template with the CORRECTED format
    content = [{"type": "text", "text": question}]
    for img_path in image_paths:
        # CORRECTED LINE: Using the proper format {"type": "image", "url": path}
        content.append({"type": "image", "url": img_path})

    # 3. Structure the conversation for the chat template
    messages = [{"role": "user", "content": content}]

    # 4. Let `apply_chat_template` do all the work
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
    except Exception as e:
        # Catching a broader range of exceptions that might occur during image loading
        print(f"Error during processor.apply_chat_template: {e}")
        return None

    # 5. Move inputs to GPU and ensure dtype consistency
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)

    # 6. Generate the response
    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
    }

    generated_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs['input_ids'].shape[1]
    response_ids = generated_ids[:, input_len:]

    response = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
        context = item["context"]
        # conbining context with question
        question = f"{context} {question}" if context else question
        options = item["options"]
        ground_truth = item["output"]

        base_response = get_model_prediction(base_model, base_processor, image_paths, question)
        if base_response:
            print("Base Model Raw Response:", base_response + "...")
            base_choice = get_option_from_response(openai_client, base_response, options)
            print(f"Judged Choice for Base Model: {base_choice}")
            if base_choice == ground_truth:
                base_model_correct += 1
        # add to questions, ask the model to explain the reasoning
        question = f"First explain your reasoning for the answer and give you final answer then. {context} {question}"
        finetuned_response = get_model_prediction(finetuned_model, finetuned_processor, image_paths, question)
        if finetuned_response:
            print("Finetuned Model Raw Response:", finetuned_response + "...")
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