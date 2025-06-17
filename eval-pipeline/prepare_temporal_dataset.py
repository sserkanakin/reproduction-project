#!/usr/bin/env python3
"""
eval-pipeline/prepare_temporal_dataset.py

This script processes a temporal ordering dataset.
For training data, it uses the OpenAI reasoning API to generate explanations.
It writes the output to `train.json` and keeps the test samples unchanged.
"""

import os
import json
import argparse
import base64
import re
import logging
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format=r'%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Helper Functions ---

def encode_image_to_base64(image_path: str) -> str | None:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None

def parse_correct_order(options_str: str, correct_option_char: str) -> list | None:
    """
    Parses the options string to find the list corresponding to the correct option.
    Example: A: [3, 5, 1, 2, 0, 4]
    """
    pattern = re.compile(f"^{re.escape(correct_option_char)}:\\s*(\\[.*?\\])", re.MULTILINE)
    match = pattern.search(options_str)
    if match:
        try:
            order_list_str = match.group(1)
            return json.loads(order_list_str)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from order string: {order_list_str}")
            return None
    logging.warning(f"Could not find option {correct_option_char} in options: {options_str}")
    return None

def get_explanation_from_openai(client: OpenAI, image_paths: list[str], correct_order: list) -> str:
    """
    Generates a step-by-step explanation for the given sequence using an OpenAI vision model.
    """
    base64_images = [encode_image_to_base64(p) for p in image_paths]
    if any(img is None for img in base64_images):
        return "Error: Could not process one or more images."

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that analyzes sequences. Your task is to explain why the given sequence is correct. "
                "Provide a concise, step-by-step explanation that justifies why this ordering is the right one."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"I have {len(image_paths)} images that represent a sequence. The images are provided out of their "
                        f"chronological order. The correct order is given by the list: {correct_order}. "
                        "Please explain why this sequence is correct in a step-by-step manner, focusing on the key indicators "
                        "that justify this ordering."
                    )
                }
            ]
        }
    ]

    for b64_image in base64_images:
        messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            }
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500
        )
        explanation = response.choices[0].message.content.strip()
        return f"Here’s a concise step-by-step explanation:\n\n{explanation}"
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return "Failed to generate explanation from the AI model due to an API error."

def process_item(item: dict, client: OpenAI, mmiu_file_dir: str) -> dict | None:
    """
    Processes a single training item from the source JSON and converts it to the target format.
    Reasoning is obtained via the OpenAI API.
    """
    original_image_paths = item.get("input_image_path", [])
    if not original_image_paths:
        logging.warning("Item has no image paths, skipping.")
        return None

    full_image_paths = [os.path.normpath(os.path.join(mmiu_file_dir, p)) for p in original_image_paths]
    correct_option_char = item.get("output")
    options_str = item.get("options")

    if not correct_option_char or not options_str:
        logging.warning("Item is missing output or options, skipping.")
        return None

    correct_order = parse_correct_order(options_str, correct_option_char)
    if correct_order is None:
        logging.warning("Could not parse the correct order for the item. Skipping.")
        return None

    logging.info(f"Processing sequence with correct order: {correct_order}")

    explanation = get_explanation_from_openai(client, full_image_paths, correct_order)
    item_id = os.path.splitext(os.path.basename(original_image_paths[0]))[0]
    human_message = (
        "<image>\n" * len(original_image_paths)
        + "\nPlease predict the order of the following pictures, and give each picture a sequential index. "
          "This index starts from 0. The larger the index, the later the order."
    )
    gpt_message = f"The correct order is {correct_order}.\n\n{explanation}"

    return {
        "id": item_id,
        "images": original_image_paths,
        "conversations": [
            {"from": "human", "value": human_message},
            {"from": "gpt", "value": gpt_message}
        ]
    }

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Prepare a temporal ordering dataset for fine-tuning.")
    parser.add_argument("--mmiu_file", required=True, help="Path to the source JSON file (e.g., to-data.json).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output files.")
    parser.add_argument("--train_size", type=int, default=1, help="Number of samples for training data.")
    args = parser.parse_args()

    logging.info("Starting dataset preparation script.")

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in .env file or environment variables.")
        return
    client = OpenAI()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(args.mmiu_file, "r") as f:
            source_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Source file not found: {args.mmiu_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from source file: {args.mmiu_file}")
        return

    train_data = source_data[:args.train_size]
    test_data = source_data[args.train_size:]
    logging.info(f"Processing {len(train_data)} train items and keeping {len(test_data)} test items unchanged.")

    mmiu_file_dir = os.path.dirname(os.path.abspath(args.mmiu_file))
    train_dataset = []

    for item in tqdm(train_data, desc="Processing train items"):
        processed_item = process_item(item, client, mmiu_file_dir)
        if processed_item:
            train_dataset.append(processed_item)

    train_output = os.path.join(args.output_dir, "train.json")
    test_output = os.path.join(args.output_dir, "test.json")
    try:
        with open(train_output, "w") as f:
            json.dump(train_dataset, f, indent=2)
        with open(test_output, "w") as f:
            json.dump(test_data, f, indent=2)
        logging.info(f"✅ Successfully created train data at: {train_output}")
        logging.info(f"✅ Successfully created test data at: {test_output}")
    except Exception as e:
        logging.error(f"Failed to write output files: {e}")

if __name__ == "__main__":
    main()