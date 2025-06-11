import os
import json
import argparse
import base64
from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# --- 0. Setup and Configuration ---
# Load environment variables from .env file (for OPENAI_API_KEY)
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a reasoning-rich fine-tuning dataset from MMIU samples.")
    parser.add_argument("--mmiu_hf_dataset_name", type=str, default="FanqingM/MMIU-Benchmark",
                        help="Name of the MMIU dataset on Hugging Face Hub.")
    parser.add_argument("--mmiu_hf_dataset_config", type=str, default=None,
                        help="Optional: Specific configuration or 'name' for the MMIU dataset on Hugging Face Hub.")
    parser.add_argument("--mmiu_hf_dataset_split", type=str, default="test",
                        help="Split to use from the Hugging Face dataset (e.g., 'test').")
    parser.add_argument("--task_name", type=str, default="temporal_ordering",
                        help="Name of the MMIU task to process.")
    parser.add_argument("--image_base_dir", type=str, required=True,
                        help="Base directory where downloaded MMIU images are stored.")
    parser.add_argument("--output_dir", type=str, default="./finetuning_data",
                        help="Directory to save the split datasets and the final fine-tuning file.")
    parser.add_argument("--train_split_size", type=int, default=150,
                        help="Number of samples to use for the training set.")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key. If not provided, attempts to use OPENAI_API_KEY from .env.")
    parser.add_argument("--openai_base_url", type=str, default=None,
                        help="OpenAI API base URL. Defaults to env var or official URL if not provided.")
    parser.add_argument("--generator_model", type=str, default="gpt-4o",
                        help="The powerful LLM to use for generating reasoning steps (e.g., 'gpt-4o', 'gpt-4-turbo').")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of training samples to generate responses for (for quick testing).")

    return parser.parse_args()


def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def build_generator_prompt(question, options):
    """Builds the prompt for the generator LLM to create a reasoning-rich response."""
    return (
        "You are an expert in analyzing sequences of images to determine temporal order. "
        "I will provide you with several images, a question, and multiple-choice options. "
        "Your task is to provide a step-by-step reasoning for determining the correct chronological order of the events shown in the images based on visual evidence. "
        "Your reasoning should be clear, logical, and reference the content of each image by its input number (Image 0, Image 1, etc.).\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options}\n\n"
        "After your reasoning, you must conclude with the final correct option letter on a new line, formatted exactly as: 'Final Answer: [LETTER]'."
    )


def generate_golden_response(client, generator_model, sample, image_base_dir):
    """Uses a powerful LLM to generate a reasoning-rich 'golden' response for a single MMIU sample."""
    if not client:
        raise ValueError("OpenAI client is not initialized.")

    image_paths = [os.path.join(image_base_dir, p) for p in sample.get("input_image_path", []) if p]

    messages = [{"role": "user", "content": []}]

    image_content = []
    for img_path in image_paths:
        base64_image = encode_image_to_base64(img_path)
        if base64_image:
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    if not image_content:
        return "Error: Could not encode any images for this sample."

    messages[0]["content"].extend(image_content)

    prompt_text = build_generator_prompt(sample.get("question", ""), sample.get("options", ""))
    messages[0]["content"].append({"type": "text", "text": prompt_text})

    try:
        response = client.chat.completions.create(
            model=generator_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while calling the OpenAI API for sample ID {sample.get('id', 'N/A')}: {e}")
        return f"Error: API call failed. {e}"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Starting dataset generation process. Output will be saved in: {args.output_dir}")

    print(f"Loading MMIU dataset '{args.mmiu_hf_dataset_name}' from Hugging Face Hub...")
    try:
        hf_dataset = load_dataset(
            args.mmiu_hf_dataset_name,
            name=args.mmiu_hf_dataset_config,
            split=args.mmiu_hf_dataset_split
        )
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load dataset from Hugging Face Hub. {e}");
        return

    task_samples = [s for s in hf_dataset if
                    isinstance(s, dict) and s.get("task", "").lower() == args.task_name.lower()]
    if not task_samples:
        print(f"Error: No samples found for task '{args.task_name}'. Exiting.");
        return

    print(f"Found {len(task_samples)} total samples for task '{args.task_name}'.")

    if len(task_samples) < args.train_split_size:
        print(
            f"Error: Requested training split size ({args.train_split_size}) is larger than the number of available samples ({len(task_samples)}).")
        return

    train_samples = task_samples[:args.train_split_size]
    test_samples = task_samples[args.train_split_size:]

    print(f"Splitting data: {len(train_samples)} samples for fine-tuning, {len(test_samples)} for held-out testing.")

    train_split_path = os.path.join(args.output_dir, 'train_split_source.json')
    test_split_path = os.path.join(args.output_dir, 'test_split_source.json')
    with open(train_split_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    with open(test_split_path, 'w') as f:
        json.dump(test_samples, f, indent=2)
    print(f"Source train/test splits saved to {train_split_path} and {test_split_path}")

    if args.max_samples is not None:
        print(f"Processing a maximum of {args.max_samples} samples for golden response generation.")
        samples_to_process = train_samples[:args.max_samples]
    else:
        samples_to_process = train_samples

    print("\nStarting generation of 'golden' reasoning responses...")
    final_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    final_base_url = args.openai_base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    if not final_api_key:
        print("Fatal Error: OpenAI API key is required. Set via --openai_api_key or .env file.");
        return

    try:
        client = OpenAI(api_key=final_api_key, base_url=final_base_url)
        client.models.list()
        print(f"OpenAI client initialized successfully for data generation using model: {args.generator_model}")
    except Exception as e:
        print(f"Fatal Error: Could not initialize OpenAI client. {e}");
        return

    finetuning_output_path = os.path.join(args.output_dir, 'finetuning_dataset.jsonl')

    with open(finetuning_output_path, 'w') as f_out:
        for sample in tqdm(samples_to_process, desc=f"Generating Golden Responses ({args.generator_model})"):
            image_placeholders = "".join(["<image>\n" for _ in sample.get("input_image_path", []) if _])
            context_text = sample.get("context", "")
            question_text = sample.get("question", "")
            options_text = sample.get("options", "")

            prompt_parts = ["USER:"]
            if image_placeholders: prompt_parts.append(image_placeholders)
            if context_text: prompt_parts.append(f"Context: {context_text}")
            prompt_parts.append(f"Question: {question_text}")
            if options_text: prompt_parts.append(f"Choices:\n{options_text}")
            prompt_parts.append("ASSISTANT:")
            llava_instruction_prompt = "\n".join(prompt_parts)

            golden_output = generate_golden_response(client, args.generator_model, sample, args.image_base_dir)

            finetuning_record = {
                "id": sample.get("id", "unknown"),
                "source_images": sample.get("input_image_path"),
                "instruction": llava_instruction_prompt,
                "output": golden_output,
                "ground_truth_option": sample.get("output")
            }

            f_out.write(json.dumps(finetuning_record) + '\n')

    print(f"\nFine-tuning dataset generation complete. Output saved to: {finetuning_output_path}")


if __name__ == '__main__':
    main()
