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
from dotenv import load_dotenv
from tqdm import tqdm  # ADDED for progress bars

# --- 0. Load Environment Variables from .env file ---
load_dotenv()


# --- Argument Parsing ---
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
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key for choice conversion. If not provided, attempts to use OPENAI_API_KEY env var (loaded from .env if present).")
    parser.add_argument("--openai_base_url", type=str, default=None,
                        help="OpenAI API base URL. If not provided, attempts to use OPENAI_BASE_URL env var, then defaults to https://api.openai.com/v1.")
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
    # ADDED: Verbosity argument
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level: 0 = silent (only final summary and errors), 1 = normal (progress bars, key steps), 2 = detailed (individual sample info).")

    return parser.parse_args()


# --- Global verbosity level ---
VERBOSITY_LEVEL = 1  # Default, will be updated by args


def log_message(message, level=1):
    """Prints message if current verbosity level is high enough."""
    if VERBOSITY_LEVEL >= level:
        print(message)


# --- 1. LLaVA Inference Module ---
llava_model = None
llava_processor = None
llava_device = None
llava_dtype = None


def initialize_llava_model(model_id, use_quantization, device_override=None):
    global llava_model, llava_processor, llava_device, llava_dtype

    if llava_model is not None:
        return

    if device_override:
        llava_device = device_override
    elif torch.cuda.is_available():
        llava_device = "cuda"
    elif torch.backends.mps.is_available():
        llava_device = "mps"
    else:
        llava_device = "cpu"
    log_message(f"Using LLaVA device: {llava_device}", 1)

    if llava_device == "cuda":
        llava_dtype = torch.float16
    else:
        llava_dtype = torch.float32
    log_message(f"Using LLaVA dtype for loading: {llava_dtype}", 1)

    model_kwargs = {
        "torch_dtype": llava_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }

    if use_quantization and llava_device != "cpu":
        log_message("Attempting to load LLaVA model in 8-bit.", 1)
        model_kwargs["load_in_8bit"] = True

    try:
        log_message(f"Loading LLaVA model: {model_id} with kwargs: {model_kwargs}", 1)
        llava_model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        if not (use_quantization and model_kwargs.get("load_in_8bit")):
            llava_model = llava_model.to(llava_device)
        llava_model.eval()
        log_message(
            f"LLaVA Model loaded. On device: {next(llava_model.parameters()).device}, Dtype: {next(llava_model.parameters()).dtype}",
            1)

        llava_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        log_message("LLaVA Processor loaded.", 1)

    except Exception as e:
        log_message(f"Error loading LLaVA model or processor for '{model_id}': {e}", 0)  # Critical error
        raise


def run_llava_single_inference(image_paths, question_with_options):
    global llava_model, llava_processor, llava_device

    if llava_model is None or llava_processor is None:
        raise RuntimeError("LLaVA model not initialized. Call initialize_llava_model first.")

    loaded_images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            log_message(f"Warning: Image file not found at '{image_path}'. Skipping image.", 1)
            return "image_path_error"
        try:
            image = Image.open(image_path).convert("RGB")
            loaded_images.append(image)
        except Exception as e:
            log_message(f"Error opening or converting image '{image_path}': {e}", 1)
            return "image_load_error"

    if not loaded_images:
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

        if 'pixel_values' in inputs and inputs['pixel_values'].dtype != llava_model.dtype:
            inputs['pixel_values'] = inputs['pixel_values'].to(llava_model.dtype)

        generation_kwargs = {"max_new_tokens": 10, "num_beams": 1, "do_sample": False}
        with torch.no_grad():
            generated_ids = llava_model.generate(**inputs, **generation_kwargs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        input_token_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        response_ids = generated_ids[:, input_token_len:]

        raw_prediction = llava_processor.batch_decode(response_ids, skip_special_tokens=True)[0]
        return raw_prediction.strip()

    except Exception as e:
        log_message(f"Error during LLaVA inference: {e}", 1)
        return "llava_model_error"


# --- 2. Choice Conversion Module ---
openai_client = None


def initialize_openai_client(api_key_arg, base_url_arg):
    global openai_client

    final_api_key = api_key_arg or os.environ.get("OPENAI_API_KEY")
    final_base_url = base_url_arg or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    if not final_api_key:
        log_message("Warning: OpenAI API key not provided. Choice conversion will use regex fallback.", 1)
        openai_client = None
        return
    try:
        openai_client = OpenAI(api_key=final_api_key, base_url=final_base_url)
        openai_client.models.list()
        source_msg = "argument" if api_key_arg else "environment variable (.env or shell)"
        log_message(
            f"OpenAI client initialized successfully (using key from: {source_msg}). Base URL: {final_base_url}", 1)
    except Exception as e:
        log_message(f"Failed to initialize OpenAI client: {e}. Choice conversion will use regex fallback.", 1)
        openai_client = None


def remove_punctuation_for_choice(text):
    return re.sub(r'^[.,()\s]+|[.,()\s]+$', '', text)


def build_choice_prompt(question, options, prediction):
    # (Prompt content remains the same)
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
    # This function is called by Pool.map, so its prints might get interleaved if not careful.
    # For simplicity, major warnings/errors from here are still printed.
    global openai_client

    sample_id = sample_with_raw_prediction.get('id', 'N/A')  # Get ID for logging
    options_text = sample_with_raw_prediction['options']
    question_text = sample_with_raw_prediction['question']
    raw_prediction = sample_with_raw_prediction.get('raw_prediction', "").strip()

    if not raw_prediction or raw_prediction.lower() in ["image_path_error", "image_load_error", "no_images_loaded",
                                                        "llava_model_error", "model error or image error", "image none",
                                                        "image error", "model error"]:
        sample_with_raw_prediction['converted_choice'] = 'Z'
        log_message(f"  Sample {sample_id}: Error in raw prediction ('{raw_prediction}'), choice set to Z.", 2)
        return sample_with_raw_prediction

    direct_choice_match = re.match(r"^\s*([A-H])\s*[.,()!?]?\s*$", raw_prediction, re.IGNORECASE)
    if direct_choice_match:
        choice = direct_choice_match.group(1).upper()
        sample_with_raw_prediction['converted_choice'] = choice
        log_message(f"  Sample {sample_id}: Direct regex match for choice '{choice}'. Raw: '{raw_prediction[:30]}...'",
                    2)
        return sample_with_raw_prediction

    if openai_client:
        cleaned_prediction = remove_punctuation_for_choice(raw_prediction)
        if not cleaned_prediction:
            sample_with_raw_prediction['converted_choice'] = 'Z'
            log_message(f"  Sample {sample_id}: Prediction empty after cleaning, choice set to Z.", 2)
            return sample_with_raw_prediction

        prompt_content = build_choice_prompt(question_text, options_text, cleaned_prediction)
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=5, temperature=0.0
            )
            grading = response.choices[0].message.content.strip().upper()
            if re.match(r"^[A-Z]$", grading):
                sample_with_raw_prediction['converted_choice'] = grading
                log_message(
                    f"  Sample {sample_id}: OpenAI converted choice to '{grading}'. Raw: '{raw_prediction[:30]}...'", 2)
            else:
                log_message(
                    f"Warning: OpenAI returned unexpected choice format: '{grading}' for sample {sample_id}. Defaulting to Z.",
                    1)
                sample_with_raw_prediction['converted_choice'] = 'Z'
        except Exception as e:
            log_message(f"Error during OpenAI call for sample {sample_id}: {e}. Defaulting to Z.", 1)
            sample_with_raw_prediction['converted_choice'] = 'Z'
    else:
        log_message(
            f"Warning: OpenAI client not available. Using basic regex for choice conversion for sample {sample_id}.", 1)
        first_char = raw_prediction[0].upper() if raw_prediction else 'Z'
        if first_char in [chr(ord('A') + i) for i in range(8)]:
            choice = first_char
            sample_with_raw_prediction['converted_choice'] = choice
            log_message(f"  Sample {sample_id}: Regex fallback choice '{choice}'. Raw: '{raw_prediction[:30]}...'", 2)
        else:
            sample_with_raw_prediction['converted_choice'] = 'Z'
            log_message(
                f"  Sample {sample_id}: Regex fallback failed, choice set to Z. Raw: '{raw_prediction[:30]}...'", 2)

    return sample_with_raw_prediction


# --- 3. Main Pipeline Orchestration ---
def main_pipeline():
    args = parse_args()
    global VERBOSITY_LEVEL
    VERBOSITY_LEVEL = args.verbose  # Set global verbosity

    os.makedirs(args.output_dir, exist_ok=True)
    log_message(f"Pipeline started. Output directory: {args.output_dir}", 1)
    log_message(f"Verbosity level set to: {VERBOSITY_LEVEL}", 1)

    # --- Stage 1: LLaVA Inference ---
    raw_predictions_file = os.path.join(args.output_dir, "raw_llava_predictions.json")

    if args.skip_llava_inference and os.path.exists(raw_predictions_file):
        log_message(f"Skipping LLaVA inference. Loading raw predictions from {raw_predictions_file}", 1)
        with open(raw_predictions_file, 'r') as f:
            all_samples_with_raw_predictions = json.load(f)
    else:
        log_message("Starting LLaVA inference stage...", 1)
        initialize_llava_model(args.llava_model_id, args.use_quantization, args.device)

        try:
            with open(args.mmiu_data_path, 'r') as f:
                mmiu_tasks_data = json.load(f)
        except FileNotFoundError:
            log_message(f"Error: MMIU data file not found at {args.mmiu_data_path}", 0)
            return
        except json.JSONDecodeError:
            log_message(f"Error: Could not decode JSON from {args.mmiu_data_path}", 0)
            return

        if not isinstance(mmiu_tasks_data, list):
            log_message(f"Error: MMIU data in {args.mmiu_data_path} is not a list of samples as expected.", 0)
            return

        temporal_ordering_samples = [s for s in mmiu_tasks_data if
                                     isinstance(s, dict) and s.get("task", "").lower() == "temporal_ordering"]

        if not temporal_ordering_samples and isinstance(mmiu_tasks_data, list) and all(
                isinstance(s, dict) and "question" in s for s in mmiu_tasks_data):
            log_message(
                "Warning: No 'task':'temporal_ordering' found, but data is a list of samples. Assuming all samples are for the target task.",
                1)
            temporal_ordering_samples = mmiu_tasks_data
        elif not temporal_ordering_samples:
            log_message(
                f"Error: No samples found for task 'temporal_ordering' or data format not as expected in {args.mmiu_data_path}",
                0)
            return

        log_message(f"Found {len(temporal_ordering_samples)} samples for temporal ordering.", 1)

        if args.max_samples is not None:
            temporal_ordering_samples = temporal_ordering_samples[:args.max_samples]
            log_message(f"Processing a maximum of {args.max_samples} samples.", 1)

        all_samples_with_raw_predictions = []
        # ADDED: tqdm progress bar for LLaVA inference
        for i, sample in enumerate(
                tqdm(temporal_ordering_samples, desc="LLaVA Inference", disable=VERBOSITY_LEVEL < 1)):
            log_message(f"Processing sample {i + 1}/{len(temporal_ordering_samples)}: ID {sample.get('id', 'N/A')}", 2)

            current_image_paths = [os.path.join(args.image_base_dir, p) for p in sample.get("input_image_path", [])]

            question_text = sample.get("question", "")
            options_text = sample.get("options", "")
            context_text = sample.get("context", "")

            prompt_for_llava = f"{context_text}\n{question_text}\n{options_text}\nPlease answer with the option letter only (e.g., A, B, C, D)."
            if not context_text:
                prompt_for_llava = f"{question_text}\n{options_text}\nPlease answer with the option letter only (e.g., A, B, C, D)."

            raw_pred = run_llava_single_inference(current_image_paths, prompt_for_llava)
            log_message(f"  LLaVA raw pred for {sample.get('id', 'N/A')}: '{raw_pred[:50]}...'", 2)

            output_sample = sample.copy()
            output_sample['raw_prediction'] = raw_pred
            all_samples_with_raw_predictions.append(output_sample)

        with open(raw_predictions_file, 'w') as f:
            json.dump(all_samples_with_raw_predictions, f, indent=2)
        log_message(f"LLaVA raw predictions saved to {raw_predictions_file}", 1)

    # --- Stage 2: Choice Conversion ---
    choice_predictions_file = os.path.join(args.output_dir, "choice_llava_predictions.json")
    if args.skip_choice_conversion and os.path.exists(choice_predictions_file):
        log_message(f"Skipping choice conversion. Loading choice predictions from {choice_predictions_file}", 1)
        with open(choice_predictions_file, 'r') as f:
            all_samples_with_choices = json.load(f)
    else:
        log_message("Starting choice conversion stage...", 1)
        initialize_openai_client(args.openai_api_key, args.openai_base_url)

        if not all_samples_with_raw_predictions:
            if os.path.exists(raw_predictions_file):
                with open(raw_predictions_file, 'r') as f:
                    all_samples_with_raw_predictions = json.load(f)
            else:
                log_message(
                    "Error: Raw predictions file not found and inference was skipped. Cannot proceed with choice conversion.",
                    0)
                return

        if not all_samples_with_raw_predictions:
            log_message("No raw predictions to process for choice conversion.", 1)
            return

        if openai_client and args.num_processes_choice_conversion > 0:
            log_message(f"Using {args.num_processes_choice_conversion} processes for OpenAI calls.", 1)
            # ADDED: tqdm for multiprocessing choice conversion
            with Pool(processes=args.num_processes_choice_conversion) as pool:
                all_samples_with_choices = list(
                    tqdm(pool.imap(process_single_choice_conversion, all_samples_with_raw_predictions),
                         total=len(all_samples_with_raw_predictions),
                         desc="Choice Conversion (OpenAI)",
                         disable=VERBOSITY_LEVEL < 1))
        else:
            log_message("Processing choice conversion in a single process (or using regex fallback).", 1)
            # ADDED: tqdm for single process choice conversion
            all_samples_with_choices = [process_single_choice_conversion(s) for s in
                                        tqdm(all_samples_with_raw_predictions, desc="Choice Conversion (Single/Regex)",
                                             disable=VERBOSITY_LEVEL < 1)]

        with open(choice_predictions_file, 'w') as f:
            json.dump(all_samples_with_choices, f, indent=2)
        log_message(f"Choice predictions saved to {choice_predictions_file}", 1)

    # --- Stage 3: Accuracy Calculation ---
    log_message("Starting accuracy calculation stage...", 1)
    if not all_samples_with_choices:
        if os.path.exists(choice_predictions_file):
            with open(choice_predictions_file, 'r') as f:
                all_samples_with_choices = json.load(f)
        else:
            log_message(
                "Error: Choice predictions file not found and conversion was skipped. Cannot calculate accuracy.", 0)
            return

    if not all_samples_with_choices:
        log_message("No choice predictions to calculate accuracy.", 1)
        return

    correct_count = 0
    total_count = 0
    error_choice_count = 0

    # ADDED: tqdm for accuracy calculation loop
    for sample in tqdm(all_samples_with_choices, desc="Calculating Accuracy", disable=VERBOSITY_LEVEL < 1):
        ground_truth = sample.get("output", "").strip().upper()
        if not ground_truth and "answer" in sample:
            ground_truth = sample.get("answer", "").strip().upper()

        predicted_choice = sample.get("converted_choice", "Z").strip().upper()

        if not ground_truth:
            log_message(
                f"Warning: Missing ground truth (checked 'output' and 'answer' keys) for sample ID {sample.get('id', 'N/A')}. Skipping.",
                1)
            continue

        total_count += 1
        if predicted_choice == ground_truth:
            correct_count += 1
        if predicted_choice == 'Z':
            error_choice_count += 1

        log_message(
            f"  Sample {sample.get('id', 'N/A')}: GT='{ground_truth}', Pred='{predicted_choice}', Correct={predicted_choice == ground_truth}",
            2)

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    summary_message = (
        "\n--- Evaluation Summary ---\n"
        f"Total Samples Evaluated: {total_count}\n"
        f"Correct Predictions: {correct_count}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Samples with 'Z' (error/unanswerable) choice: {error_choice_count}\n"
        "--------------------------"
    )
    log_message(summary_message, 0)  # Always print final summary

    summary_file = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"LLaVA Model ID: {args.llava_model_id}\n")
        f.write(f"MMIU Data Path: {args.mmiu_data_path}\n")
        f.write(summary_message)  # Write the same summary to file
    log_message(f"Evaluation summary saved to {summary_file}", 1)


if __name__ == "__main__":
    main_pipeline()
