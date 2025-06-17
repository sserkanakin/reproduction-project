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
from tqdm import tqdm
from datasets import load_dataset

load_dotenv()

VERBOSITY_LEVEL = 1


def log_message(message, level=1):
    if VERBOSITY_LEVEL >= level:
        print(message)

base_model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"  # Default model ID

def parse_args():
    parser = argparse.ArgumentParser(description="Generalized MMIU Evaluation Pipeline using Hugging Face Datasets")
    # --- MODIFIED: Changed argument handling for data source ---
    parser.add_argument("--mmiu_hf_dataset_name", type=str, default="FanqingM/MMIU-Benchmark",
                        help="Name of the MMIU dataset on Hugging Face Hub (e.g., 'OpenGVLab/MMIU').")
    parser.add_argument("--mmiu_hf_dataset_config", type=str, default=None,
                        help="Optional: Specific configuration or 'name' for the MMIU dataset on Hugging Face Hub.")
    parser.add_argument("--mmiu_hf_dataset_split", type=str, default="test",
                        help="Specific split for the MMIU dataset from Hugging Face Hub (e.g., 'test').")
    # MODIFIED: Changed mmiu_local_json_path to be optional and renamed for clarity
    parser.add_argument("--mmiu_local_json_path", type=str, default=None,
                        help="Optional: Path to a local MMIU JSON metadata file. Use this if you don't want to load from Hugging Face Hub.")

    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the specific MMIU task to evaluate (e.g., 'temporal_ordering'). Case-insensitive.")
    # MODIFIED: image_base_dir is now always required
    parser.add_argument("--image_base_dir", type=str, required=True,
                        help="Base directory where MMIU images (downloaded separately) are stored. This is combined with relative paths from the dataset.")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-interleave-qwen-0.5b-hf")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--use_quantization", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_processes_choice_conversion", type=int, default=4)
    parser.add_argument("--skip_model_inference", action="store_true")
    parser.add_argument("--skip_choice_conversion", action="store_true")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])

    args = parser.parse_args()

    # --- MODIFIED: Added validation logic ---
    if not args.mmiu_hf_dataset_name and not args.mmiu_local_json_path:
        parser.error(
            "You must provide either --mmiu_hf_dataset_name to load from Hugging Face Hub or --mmiu_local_json_path to load from a local file.")

    return args


# ... (The rest of your script from the previous version remains the same) ...
# Globals for the current model
current_model = None
current_processor = None
current_model_device = None
current_model_dtype = None


def initialize_model(model_id_arg, use_quantization, device_override=None):
    global current_model, current_processor, current_model_device, current_model_dtype
    if current_model is not None: return
    if device_override:
        current_model_device = device_override
    elif torch.cuda.is_available():
        current_model_device = "cuda"
    elif torch.backends.mps.is_available():
        current_model_device = "mps"
    else:
        current_model_device = "cpu"
    log_message(f"Using Model device: {current_model_device}", 1)
    if current_model_device == "cuda":
        current_model_dtype = torch.float16
    else:
        current_model_dtype = torch.float32
    log_message(f"Using Model dtype for loading: {current_model_dtype}", 1)
    model_kwargs = {"torch_dtype": current_model_dtype, "low_cpu_mem_usage": True, "trust_remote_code": True}
    if use_quantization and current_model_device != "cpu":
        log_message(f"Attempting to load model {model_id_arg} in 8-bit.", 1);
        model_kwargs["load_in_8bit"] = True
    try:
        log_message(f"Loading model: {model_id_arg} with kwargs: {model_kwargs}", 1)
        current_model = LlavaForConditionalGeneration.from_pretrained(model_id_arg, **model_kwargs)
        if not (use_quantization and model_kwargs.get("load_in_8bit")): current_model = current_model.to(
            current_model_device)
        current_model.eval()
        log_message(
            f"Model loaded. On device: {next(current_model.parameters()).device}, Dtype: {next(current_model.parameters()).dtype}",
            1)
        current_processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        log_message("Processor loaded.", 1)
    except Exception as e:
        log_message(f"Error loading model or processor for '{model_id_arg}': {e}", 0); raise


def run_model_single_inference(image_full_paths, full_prompt_for_model):
    global current_model, current_processor, current_model_device
    if current_model is None or current_processor is None: raise RuntimeError("Model not initialized.")
    loaded_images = []
    if image_full_paths:
        for image_path in image_full_paths:
            if not os.path.exists(image_path): log_message(f"Warning: Image file not found: '{image_path}'.",
                                                           1); return "image_path_error"
            try:
                image = Image.open(image_path).convert("RGB"); loaded_images.append(image)
            except Exception as e:
                log_message(f"Error opening image '{image_path}': {e}", 1); return "image_load_error"
    if "<image>" in full_prompt_for_model.lower() and not loaded_images and image_full_paths:
        log_message(f"Warning: Prompt expects images, but none loaded.", 1);
        return "no_images_loaded_for_image_prompt"
    try:
        inputs = current_processor(text=full_prompt_for_model, images=loaded_images if loaded_images else None,
                                   return_tensors="pt").to(current_model_device)
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None and inputs[
            'pixel_values'].dtype != current_model.dtype:
            inputs['pixel_values'] = inputs['pixel_values'].to(current_model.dtype)
        generation_kwargs = {"max_new_tokens": 128, "num_beams": 1, "do_sample": False}
        with torch.no_grad():
            generated_ids = current_model.generate(**inputs, **generation_kwargs)
        input_token_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        response_ids = generated_ids[:, input_token_len:]
        raw_prediction = current_processor.batch_decode(response_ids, skip_special_tokens=True)[0]
        return raw_prediction.strip()
    except Exception as e:
        log_message(f"Error during model inference: {e}", 1); return "model_inference_error"


openai_client = None


def initialize_openai_client(api_key_arg, base_url_arg):
    global openai_client;
    final_api_key = api_key_arg or os.environ.get("OPENAI_API_KEY")
    final_base_url = base_url_arg or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if not final_api_key: log_message("Warning: OpenAI API key not set. Regex fallback for choice conversion.",
                                      1); openai_client = None; return
    try:
        openai_client = OpenAI(api_key=final_api_key, base_url=final_base_url);
        openai_client.models.list()
        source_msg = "argument" if api_key_arg else "env";
        log_message(f"OpenAI client initialized (key from: {source_msg}). URL: {final_base_url}", 1)
    except Exception as e:
        log_message(f"Failed to initialize OpenAI client: {e}. Regex fallback.", 1); openai_client = None


def remove_punctuation_for_choice(text): return re.sub(r'^[.,()\s]+|[.,()\s]+$', '', text)


def build_choice_prompt(q, o, p):
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character corresponding to the option letter (e.g., A, B, C, D). "
        "Do not provide any explanation or other text. Only the single character. \n"
        "Example 1: \nQuestion: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: a cute teddy bear\nYour output: A\n"
        "Example 2: \nQuestion: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: Spider\nYour output: Z\n"
        "Example 3: \nQuestion: {}?\nOptions: {}\nAnswer: {}\nYour output: ")
    return tmpl.format(q, o, p)


def process_single_choice_conversion(sample_with_raw_prediction):
    global openai_client;
    sample_id = sample_with_raw_prediction.get('id', 'N/A')
    options_text = sample_with_raw_prediction.get('options', "");
    question_text = sample_with_raw_prediction.get('question', "")
    raw_prediction = sample_with_raw_prediction.get('raw_prediction', "").strip()
    error_strings = ["image_path_error", "image_load_error", "no_images_loaded", "model_inference_error",
                     "model error or image error", "image none", "image error", "model error",
                     "no_images_loaded_for_image_prompt"]
    if not raw_prediction or raw_prediction.lower() in error_strings:
        sample_with_raw_prediction['converted_choice'] = 'Z';
        log_message(f"  Sample {sample_id}: Error in raw pred ('{raw_prediction}'), choice Z.", 2);
        return sample_with_raw_prediction
    if not options_text:
        log_message(
            f"Warning: No options for sample {sample_id} (Task: {sample_with_raw_prediction.get('task')}). Storing raw.",
            1)
        sample_with_raw_prediction['converted_choice'] = raw_prediction;
        return sample_with_raw_prediction
    direct_choice_match = re.match(r"^\s*([A-H])\s*[.,()!?]?\s*$", raw_prediction, re.IGNORECASE)
    if direct_choice_match:
        choice = direct_choice_match.group(1).upper();
        sample_with_raw_prediction['converted_choice'] = choice
        log_message(f"  Sample {sample_id}: Direct regex match '{choice}'.", 2);
        return sample_with_raw_prediction
    if openai_client:
        cleaned_prediction = remove_punctuation_for_choice(raw_prediction)
        if not cleaned_prediction: sample_with_raw_prediction['converted_choice'] = 'Z'; log_message(
            f"  Sample {sample_id}: Pred empty after cleaning, choice Z.", 2); return sample_with_raw_prediction
        prompt_content = build_choice_prompt(question_text, options_text, cleaned_prediction)
        try:
            response = openai_client.chat.completions.create(model="gpt-4o-mini",
                                                             messages=[{"role": "user", "content": prompt_content}],
                                                             max_tokens=5, temperature=0.0)
            grading = response.choices[0].message.content.strip().upper()
            if re.match(r"^[A-Z]$", grading):
                sample_with_raw_prediction['converted_choice'] = grading; log_message(
                    f"  Sample {sample_id}: OpenAI choice '{grading}'.", 2)
            else:
                log_message(f"Warning: OpenAI unexpected format: '{grading}' for {sample_id}. Default Z.", 1);
                sample_with_raw_prediction['converted_choice'] = 'Z'
        except Exception as e:
            log_message(f"Error OpenAI call {sample_id}: {e}. Default Z.", 1); sample_with_raw_prediction[
                'converted_choice'] = 'Z'
    else:
        log_message(f"Warning: OpenAI client NA. Basic regex for {sample_id}.", 1)
        first_char = raw_prediction[0].upper() if raw_prediction else 'Z'
        if first_char in [chr(ord('A') + i) for i in range(8)]:
            sample_with_raw_prediction['converted_choice'] = first_char; log_message(
                f"  Sample {sample_id}: Regex fallback '{first_char}'.", 2)
        else:
            sample_with_raw_prediction['converted_choice'] = 'Z'; log_message(
                f"  Sample {sample_id}: Regex fallback failed, choice Z.", 2)
    return sample_with_raw_prediction


def main_pipeline():
    args = parse_args()
    global VERBOSITY_LEVEL;
    VERBOSITY_LEVEL = args.verbose
    model_name_slug = args.model_id.split('/')[-1].replace('-', '_').replace('.', '_')
    task_name_slug = args.task_name.lower().replace(' ', '_').replace('/', '_')
    run_specific_output_dir = os.path.join(args.output_dir, f"{model_name_slug}_{task_name_slug}")
    os.makedirs(run_specific_output_dir, exist_ok=True)
    log_message(
        f"Pipeline started. Task: '{args.task_name}'. Model: '{args.model_id}'. Output to: {run_specific_output_dir}",
        1)

    raw_predictions_file = os.path.join(run_specific_output_dir, "raw_model_predictions.json")
    all_samples_with_raw_predictions = []
    if args.skip_model_inference and os.path.exists(raw_predictions_file):
        log_message(f"Skipping model inference. Loading from {raw_predictions_file}", 1);
        with open(raw_predictions_file, 'r') as f:
            all_samples_with_raw_predictions = json.load(f)
    else:
        log_message(f"Starting model inference stage for task: {args.task_name}", 1)
        initialize_model(args.model_id, args.use_quantization, args.device)
        # --- MODIFIED: Data Loading Logic ---
        all_mmiu_data_iterable = None
        if args.mmiu_hf_dataset_name and not args.mmiu_local_json_path:  # Prioritize HF if specified
            try:
                log_message(f"Attempting to load from Hugging Face Hub: '{args.mmiu_hf_dataset_name}'...", 1)
                hf_dataset = load_dataset(args.mmiu_hf_dataset_name, name=args.mmiu_hf_dataset_config,
                                          split=args.mmiu_hf_dataset_split)
                all_mmiu_data_iterable = hf_dataset
                log_message(f"Successfully loaded {len(all_mmiu_data_iterable)} samples from Hugging Face Hub.", 1)
            except Exception as e:
                log_message(f"Error loading from Hugging Face Hub: {e}", 0);
                log_message(f"Ensure the dataset name '{args.mmiu_hf_dataset_name}' is correct.", 0);
                return
        elif args.mmiu_local_json_path:  # Fallback or explicit choice
            log_message(f"Loading from local JSON file: {args.mmiu_local_json_path}", 1)
            try:
                with open(args.mmiu_local_json_path, 'r') as f:
                    all_mmiu_data_iterable = json.load(f)
                if not isinstance(all_mmiu_data_iterable, list):
                    log_message(f"Error: Data in {args.mmiu_local_json_path} is not a list.", 0);
                    return
                log_message(f"Successfully loaded {len(all_mmiu_data_iterable)} samples from local JSON.", 1)
            except Exception as e:
                log_message(f"Error loading local JSON file: {e}", 0); return
        else:  # No data source provided
            log_message("Error: No data source provided. Use --mmiu_hf_dataset_name or --mmiu_local_json_path.", 0);
            return

        target_task_samples = [s for s in all_mmiu_data_iterable if
                               isinstance(s, dict) and s.get("task", "").lower() == args.task_name.lower()]
        if not target_task_samples:
            log_message(f"Error: No samples found for task '{args.task_name}'.", 0)
            available_tasks = sorted(list(set([s.get("task", "").lower() for s in all_mmiu_data_iterable if
                                               isinstance(s, dict) and "task" in s])))
            log_message(
                f"Available tasks (first 20): {available_tasks[:20]}" + ("..." if len(available_tasks) > 20 else ""), 1)
            return
        log_message(f"Found {len(target_task_samples)} samples for task '{args.task_name}'.", 1)
        if args.max_samples is not None:
            target_task_samples = target_task_samples[:args.max_samples]
            log_message(f"Processing a maximum of {args.max_samples} samples.", 1)
        for i, sample in enumerate(
                tqdm(target_task_samples, desc=f"Model Inference ({args.task_name})", disable=VERBOSITY_LEVEL < 1)):
            log_message(f"Processing sample {i + 1}/{len(target_task_samples)}: ID {sample.get('id', 'N/A')}", 2)
            current_image_paths_from_sample = sample.get("input_image_path", [])
            current_image_full_paths = []
            if not args.image_base_dir: log_message(
                f"Error: --image_base_dir is required for sample ID {sample.get('id', 'N/A')}.", 0); return
            if isinstance(current_image_paths_from_sample, list):
                current_image_full_paths = [os.path.join(args.image_base_dir, p) for p in
                                            current_image_paths_from_sample if p and isinstance(p, str)]
            elif isinstance(current_image_paths_from_sample, str) and current_image_paths_from_sample:
                current_image_full_paths = [os.path.join(args.image_base_dir, current_image_paths_from_sample)]
            images_prompt_section = "".join(
                ["<image>\n" for _ in current_image_full_paths]) if current_image_full_paths else ""
            context_text = sample.get("context", "");
            question_text = sample.get("question", "");
            options_text = sample.get("options", "")
            prompt_parts = ["USER:"];
            if images_prompt_section: prompt_parts.append(images_prompt_section)
            if context_text: prompt_parts.append(f"Context: {context_text}")
            prompt_parts.append(f"Question: {question_text}")
            if options_text: prompt_parts.append(f"Choices:\n{options_text}")
            prompt_parts.append("Hint: Please answer the option directly like A, B, C, D...");
            prompt_parts.append("ASSISTANT:")
            full_prompt_for_model = "\n".join(prompt_parts)
            log_message(f"  Prompt for model (ID {sample.get('id', 'N/A')}):\n{full_prompt_for_model[:1000]}...", 2)
            raw_pred = run_model_single_inference(current_image_full_paths, full_prompt_for_model)
            log_message(f"  Model raw pred for {sample.get('id', 'N/A')}: '{raw_pred[:100]}...'", 2)
            output_sample = dict(sample)
            output_sample['raw_prediction'] = raw_pred;
            output_sample['model_prompt'] = full_prompt_for_model
            all_samples_with_raw_predictions.append(output_sample)
        with open(raw_predictions_file, 'w') as f:
            json.dump(all_samples_with_raw_predictions, f, indent=2)
        log_message(f"Model raw predictions saved to {raw_predictions_file}", 1)

    # --- Stage 2: Choice Conversion ---
    choice_predictions_file = os.path.join(run_specific_output_dir, "choice_model_predictions.json")
    if args.skip_choice_conversion and os.path.exists(choice_predictions_file):
        log_message(f"Skipping choice conversion. Loading from {choice_predictions_file}", 1)
        with open(choice_predictions_file, 'r') as f:
            all_samples_with_choices = json.load(f)
    else:
        log_message("Starting choice conversion stage...", 1);
        initialize_openai_client(args.openai_api_key, args.openai_base_url)
        if not all_samples_with_raw_predictions:
            if os.path.exists(raw_predictions_file):
                with open(raw_predictions_file, 'r') as f:
                    all_samples_with_raw_predictions = json.load(f)
            else:
                log_message("Error: Raw predictions file not found. Cannot proceed.", 0); return
        if not all_samples_with_raw_predictions: log_message("No raw predictions to process.", 1); return
        desc_choice = f"Choice Conversion ({args.task_name}, OpenAI)" if openai_client else f"Choice Conv. ({args.task_name}, Regex)"
        if openai_client and args.num_processes_choice_conversion > 0:
            log_message(f"Using {args.num_processes_choice_conversion} processes for OpenAI calls.", 1)
            with Pool(processes=args.num_processes_choice_conversion) as pool:
                all_samples_with_choices = list(
                    tqdm(pool.imap(process_single_choice_conversion, all_samples_with_raw_predictions),
                         total=len(all_samples_with_raw_predictions), desc=desc_choice, disable=VERBOSITY_LEVEL < 1))
        else:
            log_message("Processing choice conversion in a single process (or using regex fallback).", 1)
            all_samples_with_choices = [process_single_choice_conversion(s) for s in
                                        tqdm(all_samples_with_raw_predictions, desc=desc_choice,
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
            log_message("Error: Choice predictions file not found. Cannot calculate accuracy.", 0); return
    if not all_samples_with_choices: log_message("No choice predictions to calculate accuracy.", 1); return
    correct_count = 0;
    total_count = 0;
    error_choice_count = 0
    for sample in tqdm(all_samples_with_choices, desc=f"Calculating Accuracy ({args.task_name})",
                       disable=VERBOSITY_LEVEL < 1):
        ground_truth = sample.get("output", "").strip().upper()
        predicted_choice = sample.get("converted_choice", "Z").strip().upper()
        if not ground_truth: log_message(
            f"Warning: Missing ground truth for sample ID {sample.get('id', 'N/A')}. Skipping.", 1); continue
        total_count += 1
        if predicted_choice == ground_truth: correct_count += 1
        if predicted_choice == 'Z': error_choice_count += 1
        log_message(
            f"  Sample {sample.get('id', 'N/A')}: GT='{ground_truth}', Pred='{predicted_choice}', Correct={predicted_choice == ground_truth}",
            2)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    summary_message = (f"\n--- Evaluation Summary for Task: {args.task_name}, Model: {args.model_id} ---\n"
                       f"Total Samples Evaluated: {total_count}\nCorrect Predictions: {correct_count}\n"
                       f"Accuracy: {accuracy:.2f}%\nSamples with 'Z' choice: {error_choice_count}\n--------------------------")
    log_message(summary_message, 0)
    summary_file = os.path.join(run_specific_output_dir, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(
            f"Model ID: {args.model_id}\nMMIU Dataset Source: {'Hugging Face (' + args.mmiu_hf_dataset_name + ')' if args.mmiu_hf_dataset_name and not args.mmiu_local_json_path else 'Local JSON (' + str(args.mmiu_local_json_path) + ')'}\nTask Name: {args.task_name}\n")
        f.write(summary_message)
    log_message(f"Evaluation summary saved to {summary_file}", 1)


if __name__ == "__main__":
    main_pipeline()