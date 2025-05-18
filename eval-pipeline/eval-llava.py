import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import json
import re
from openai import OpenAI
from multiprocessing import Pool
import argparse
import time


# --- 0. Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="MMIU LLaVA Evaluation Pipeline")
    parser.add_argument("--mmiu_data_path", type=str, required=True,
                        help="Path to the MMIU JSON file containing temporal ordering task data.")
    parser.add_argument("--image_base_dir", type=str, required=True,
                        help="Base directory where MMIU images are stored.")
    parser.add_argument("--llava_model_id", type=str, default="llava-hf/llava-interleave-qwen-0.5b-hf",
                        help="Hugging Face Model ID for LLaVA-Interleave (or local path).")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save intermediate and final results.")
    parser.add_argument("--openai_api_key", type=str, default="YOUR_OPENAI_API_KEY",
                        help="OpenAI API key for choice conversion. Can also be set via OPENAI_API_KEY env var.")
    parser.add_argument("--openai_base_url", type=str, default="https://api.openai.com/v1",
                        help="OpenAI API base URL (if using a proxy).")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Enable 8-bit quantization for the LLaVA model (requires bitsandbytes).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run LLaVA on (e.g., 'cuda', 'mps', 'cpu'). Auto-detected if None.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for quick testing).")
    parser.add_argument("--num_processes_choice_conversion", type=int, default=4,
                        help="Number of parallel processes for choice conversion using OpenAI.")
    parser.add_argument("--skip_llava_inference", action="store_true",
                        help="Skip LLaVA inference and use existing raw_predictions.json.")
    parser.add_argument("--skip_choice_conversion", action="store_true",
                        help="Skip choice conversion and use existing choice_predictions.json.")

    return parser.parse_args()


# --- 1. LLaVA Inference Module ---
# Global model and processor to avoid reloading for each sample
llava_model = None
llava_processor = None
llava_device = None
llava_dtype = None


def initialize_llava_model(model_id, use_quantization, device_override=None):
    global llava_model, llava_processor, llava_device, llava_dtype

    if llava_model is not None:  # Already initialized
        return

    if device_override:
        llava_device = device_override
    elif torch.cuda.is_available():
        llava_device = "cuda"
    elif torch.backends.mps.is_available():
        llava_device = "mps"
    else:
        llava_device = "cpu"
    print(f"Using LLaVA device: {llava_device}")

    if llava_device == "cuda":
        llava_dtype = torch.float16
    else:  # MPS and CPU
        llava_dtype = torch.float32
    print(f"Using LLaVA dtype for loading: {llava_dtype}")

    model_kwargs = {
        "torch_dtype": llava_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }

    if use_quantization and llava_device != "cpu":
        print("Attempting to load LLaVA model in 8-bit.")
        model_kwargs["load_in_8bit"] = True
        # For 8-bit, compute dtype is often float16, even if loading dtype was different
        # model_kwargs["torch_dtype"] = torch.float16

    try:
        print(f"Loading LLaVA model: {model_id} with kwargs: {model_kwargs}")
        llava_model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        if not (use_quantization and model_kwargs.get("load_in_8bit")):
            llava_model = llava_model.to(llava_device)
        llava_model.eval()
        print(
            f"LLaVA Model loaded. On device: {next(llava_model.parameters()).device}, Dtype: {next(llava_model.parameters()).dtype}")

        llava_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("LLaVA Processor loaded.")

    except Exception as e:
        print(f"Error loading LLaVA model or processor for '{model_id}': {e}")
        raise


def run_llava_single_inference(image_paths, question_with_options):
    global llava_model, llava_processor, llava_device

    if llava_model is None or llava_processor is None:
        raise RuntimeError("LLaVA model not initialized. Call initialize_llava_model first.")

    loaded_images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at '{image_path}'. Skipping image.")
            # Depending on how LLaVA handles missing images, you might return an error or empty list
            return "image_path_error"
        try:
            image = Image.open(image_path).convert("RGB")
            loaded_images.append(image)
        except Exception as e:
            print(f"Error opening or converting image '{image_path}': {e}")
            return "image_load_error"

    if not loaded_images:  # Should ideally be caught by image_path_error
        return "no_images_loaded"

    content_parts = []
    for _ in loaded_images:
        content_parts.append({"type": "image"})
    content_parts.append({"type": "text", "text": question_with_options})
    messages = [{"role": "user", "content": content_parts}]

    try:
        prompt_text = llava_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = llava_processor(
            text=prompt_text, images=loaded_images, return_tensors="pt"
        ).to(llava_device)

        # Ensure input tensor dtypes are compatible with the model
        if 'pixel_values' in inputs and inputs['pixel_values'].dtype != llava_model.dtype:
            inputs['pixel_values'] = inputs['pixel_values'].to(llava_model.dtype)
        if 'input_ids' in inputs and inputs['input_ids'].dtype != torch.long:  # input_ids should be long
            pass  # usually handled by tokenizer

        generation_kwargs = {"max_new_tokens": 50, "num_beams": 1, "do_sample": False}
        with torch.no_grad():
            generated_ids = llava_model.generate(**inputs, **generation_kwargs)

        input_token_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        response_ids = generated_ids[:, input_token_len:]

        raw_prediction = llava_processor.batch_decode(response_ids, skip_special_tokens=True)[0]
        return raw_prediction.strip()

    except Exception as e:
        print(f"Error during LLaVA inference: {e}")
        return "llava_model_error"


# --- 2. Choice Conversion Module (Adapted from user's evaluate.py) ---
openai_client = None


def initialize_openai_client(api_key, base_url):
    global openai_client
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        print("Warning: OpenAI API key not provided or is placeholder. Choice conversion will use regex fallback.")
        openai_client = None
        return
    try:
        openai_client = OpenAI(api_key=api_key, base_url=base_url)
        # Test connection (optional, but good for early failure)
        openai_client.models.list()
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}. Choice conversion will use regex fallback.")
        openai_client = None


def remove_punctuation_for_choice(text):
    return re.sub(r'^[.,()\s]+|[.,()\s]+$', '', text)  # Also strip leading/trailing whitespace


def build_choice_prompt(question, options, prediction):
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character corresponding to the option letter (e.g., A, B, C, D). "
        "Do not provide any explanation or other text. Only the single character. \n"
        "Example 1: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: a cute teddy bear\nYour output: A\n"
        "Example 2: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: Spider\nYour output: Z\n"
        "Example 3: \n"
        "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)


def process_single_choice_conversion(sample_with_raw_prediction):
    global openai_client
    # sample_with_raw_prediction is expected to be a dict containing
    # 'question', 'options', and 'raw_prediction' (the key for raw_prediction might vary)
    # For this pipeline, let's assume 'raw_prediction' is the key.

    options_text = sample_with_raw_prediction['options']
    question_text = sample_with_raw_prediction['question']
    raw_prediction = sample_with_raw_prediction.get('raw_prediction', "").strip()

    # Handle common error/empty predictions from LLaVA first
    if not raw_prediction or raw_prediction.lower() in ["image_path_error", "image_load_error", "no_images_loaded",
                                                        "llava_model_error", "model error or image error", "image none",
                                                        "image error", "model error"]:
        sample_with_raw_prediction['converted_choice'] = 'Z'
        return sample_with_raw_prediction

    # Simple regex check: if the model already outputted a single letter A-H (common for MCQs)
    # This can save API calls and handle simple cases.
    direct_choice_match = re.match(r"^\s*([A-H])\s*[.,()!?]?\s*$", raw_prediction, re.IGNORECASE)
    if direct_choice_match:
        sample_with_raw_prediction['converted_choice'] = direct_choice_match.group(1).upper()
        return sample_with_raw_prediction

    # If OpenAI client is available and initialized, use it
    if openai_client:
        # Clean prediction slightly before sending to OpenAI
        cleaned_prediction = remove_punctuation_for_choice(raw_prediction)
        if not cleaned_prediction:  # If prediction becomes empty after cleaning
            sample_with_raw_prediction['converted_choice'] = 'Z'
            return sample_with_raw_prediction

        prompt_content = build_choice_prompt(question_text, options_text, cleaned_prediction)
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Or another preferred model like gpt-3.5-turbo
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=5,  # A single character is expected
                temperature=0.0  # For deterministic output
            )
            grading = response.choices[0].message.content.strip().upper()
            # Validate if grading is a single uppercase letter A-Z
            if re.match(r"^[A-Z]$", grading):
                sample_with_raw_prediction['converted_choice'] = grading
            else:  # If GPT outputs something unexpected
                print(
                    f"Warning: OpenAI returned unexpected choice format: '{grading}'. Defaulting to Z for sample ID {sample_with_raw_prediction.get('id', 'N/A')}")
                sample_with_raw_prediction['converted_choice'] = 'Z'  # Fallback for unexpected OpenAI output
        except Exception as e:
            print(
                f"Error during OpenAI call for sample ID {sample_with_raw_prediction.get('id', 'N/A')}: {e}. Defaulting to Z.")
            sample_with_raw_prediction['converted_choice'] = 'Z'  # Fallback for OpenAI API error
    else:
        # Fallback to regex if OpenAI client is not available
        # This is a very basic fallback, might need refinement
        print(
            f"Warning: OpenAI client not available. Using basic regex for choice conversion for sample ID {sample_with_raw_prediction.get('id', 'N/A')}.")
        first_char = raw_prediction[0].upper() if raw_prediction else 'Z'
        if first_char in [chr(ord('A') + i) for i in range(8)]:  # A-H
            sample_with_raw_prediction['converted_choice'] = first_char
        else:
            sample_with_raw_prediction['converted_choice'] = 'Z'

    return sample_with_raw_prediction


# --- 3. Main Pipeline Orchestration ---
def main_pipeline():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Stage 1: LLaVA Inference ---
    raw_predictions_file = os.path.join(args.output_dir, "raw_llava_predictions.json")

    if args.skip_llava_inference and os.path.exists(raw_predictions_file):
        print(f"Skipping LLaVA inference. Loading raw predictions from {raw_predictions_file}")
        with open(raw_predictions_file, 'r') as f:
            all_samples_with_raw_predictions = json.load(f)
    else:
        print("Starting LLaVA inference stage...")
        initialize_llava_model(args.llava_model_id, args.use_quantization, args.device)

        try:
            with open(args.mmiu_data_path, 'r') as f:
                mmiu_tasks = json.load(f)
        except FileNotFoundError:
            print(f"Error: MMIU data file not found at {args.mmiu_data_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.mmiu_data_path}")
            return

        # Filter for "temporal_ordering" task (or make this configurable)
        # Assuming MMIU data is a list of task samples, each with a "task" key.
        # If your JSON is structured differently (e.g., dict of tasks), adjust this.
        temporal_ordering_samples = [s for s in mmiu_tasks if s.get("task", "").lower() == "temporal_ordering"]
        if not temporal_ordering_samples:
            # Fallback if "task" key is not present or data is just a list of samples for this task
            if isinstance(mmiu_tasks, list) and all("question" in s for s in mmiu_tasks):
                print("Assuming all samples in the MMIU data file are for the target task.")
                temporal_ordering_samples = mmiu_tasks
            else:
                print(
                    f"Error: No samples found for task 'temporal_ordering' or data format not as expected in {args.mmiu_data_path}")
                return

        print(f"Found {len(temporal_ordering_samples)} samples for temporal ordering.")

        if args.max_samples is not None:
            temporal_ordering_samples = temporal_ordering_samples[:args.max_samples]
            print(f"Processing a maximum of {args.max_samples} samples.")

        all_samples_with_raw_predictions = []
        for i, sample in enumerate(temporal_ordering_samples):
            print(f"Processing sample {i + 1}/{len(temporal_ordering_samples)}: ID {sample.get('id', 'N/A')}")

            # Construct image paths
            # Assuming 'input_image_path' contains relative paths or just filenames
            current_image_paths = [os.path.join(args.image_base_dir, p) for p in sample.get("input_image_path", [])]

            question_text = sample.get("question", "")
            options_text = sample.get("options", "")
            context_text = sample.get("context", "")  # From your MMIU scripts, context might be used

            # Construct the full question with options, similar to MMIU scripts
            # The MMIU scripts had logic:
            # if task_data['task'] in tasks_exist: question = question + '\n' + context
            # else: question = context + '\n' + question
            # question = question + '\nPlease answer the option directly like A,B,C,D...'
            # For simplicity here, let's assume a fixed way to combine them or that the question field is pre-formatted

            prompt_for_llava = f"{context_text}\n{question_text}\n{options_text}\nPlease answer with the option letter only (e.g., A, B, C, D)."
            if not context_text:  # If context is empty, avoid leading newline
                prompt_for_llava = f"{question_text}\n{options_text}\nPlease answer with the option letter only (e.g., A, B, C, D)."

            raw_pred = run_llava_single_inference(current_image_paths, prompt_for_llava)

            # Store all necessary info for the next stages
            output_sample = sample.copy()  # Start with original sample data
            output_sample['raw_prediction'] = raw_pred
            all_samples_with_raw_predictions.append(output_sample)

            # Optional: print progress
            if (i + 1) % 10 == 0:
                print(
                    f"  ... LLaVA raw prediction for sample {sample.get('id', 'N/A')}: {raw_pred[:100]}")  # Print first 100 chars

        with open(raw_predictions_file, 'w') as f:
            json.dump(all_samples_with_raw_predictions, f, indent=2)
        print(f"LLaVA raw predictions saved to {raw_predictions_file}")

    # --- Stage 2: Choice Conversion ---
    choice_predictions_file = os.path.join(args.output_dir, "choice_llava_predictions.json")
    if args.skip_choice_conversion and os.path.exists(choice_predictions_file):
        print(f"Skipping choice conversion. Loading choice predictions from {choice_predictions_file}")
        with open(choice_predictions_file, 'r') as f:
            all_samples_with_choices = json.load(f)
    else:
        print("Starting choice conversion stage...")
        initialize_openai_client(args.openai_api_key, args.openai_base_url)

        if not all_samples_with_raw_predictions:
            print("No raw predictions to process for choice conversion.")
            return

        # Use multiprocessing for OpenAI calls if client is available
        if openai_client and args.num_processes_choice_conversion > 0:
            print(f"Using {args.num_processes_choice_conversion} processes for OpenAI calls.")
            with Pool(processes=args.num_processes_choice_conversion) as pool:
                all_samples_with_choices = pool.map(process_single_choice_conversion, all_samples_with_raw_predictions)
        else:  # Single process or regex fallback
            print("Processing choice conversion in a single process (or using regex fallback).")
            all_samples_with_choices = [process_single_choice_conversion(s) for s in all_samples_with_raw_predictions]

        with open(choice_predictions_file, 'w') as f:
            json.dump(all_samples_with_choices, f, indent=2)
        print(f"Choice predictions saved to {choice_predictions_file}")

    # --- Stage 3: Accuracy Calculation ---
    print("Starting accuracy calculation stage...")
    if not all_samples_with_choices:
        print("No choice predictions to calculate accuracy.")
        return

    correct_count = 0
    total_count = 0
    error_choice_count = 0

    for sample in all_samples_with_choices:
        # Assuming ground truth is stored with key like 'answer' or 'ground_truth_choice'
        # Adjust key if necessary based on your MMIU data format
        ground_truth = sample.get("answer", "").strip().upper()
        predicted_choice = sample.get("converted_choice", "Z").strip().upper()

        if not ground_truth:
            print(f"Warning: Missing ground truth for sample ID {sample.get('id', 'N/A')}. Skipping.")
            continue

        total_count += 1
        if predicted_choice == ground_truth:
            correct_count += 1
        if predicted_choice == 'Z':  # Count how many were marked as error/unanswerable by conversion
            error_choice_count += 1

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    print("\n--- Evaluation Summary ---")
    print(f"Total Samples Evaluated: {total_count}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Samples with 'Z' (error/unanswerable) choice: {error_choice_count}")
    print("--------------------------")

    summary_file = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"LLaVA Model ID: {args.llava_model_id}\n")
        f.write(f"MMIU Data Path: {args.mmiu_data_path}\n")
        f.write(f"Total Samples Evaluated: {total_count}\n")
        f.write(f"Correct Predictions: {correct_count}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Samples with 'Z' choice: {error_choice_count}\n")
    print(f"Evaluation summary saved to {summary_file}")


if __name__ == "__main__":
    # Example of how to set OPENAI_API_KEY if not passed as arg
    # if "OPENAI_API_KEY" not in os.environ:
    #     os.environ["OPENAI_API_KEY"] = "your_fallback_key_here_if_any"
    main_pipeline()
