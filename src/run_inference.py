import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os

# --- 1. Configuration ---

# OPTION 1: Try the smaller 0.5B parameter model for Mac testing (Recommended for memory constraints)
MODEL_ID_SMALL = "llava-hf/llava-interleave-qwen-0.5b-hf"
# OPTION 2: The original 7B parameter model (might require quantization on Mac)
MODEL_ID_7B = "llava-hf/llava-interleave-qwen-7b-hf"

# --- CHOOSE YOUR MODEL AND LOADING STRATEGY ---
# For Mac testing with limited memory, start with MODEL_ID_SMALL.
# If you want to try the 7B model on Mac, set USE_QUANTIZATION_IF_APPLICABLE to True.
# On a powerful GPU machine, you can use MODEL_ID_7B and set USE_QUANTIZATION_IF_APPLICABLE to False.

CURRENT_MODEL_ID = MODEL_ID_7B  # TRY THIS FIRST ON YOUR MAC
# CURRENT_MODEL_ID = MODEL_ID_7B

# Quantization (primarily for reducing memory on Mac for the 7B model)
# bitsandbytes library is required: pip install bitsandbytes
# Note: 8-bit is more likely to be stable on MPS than 4-bit.
# For CUDA, both 8-bit and 4-bit are well-supported.
USE_QUANTIZATION_IF_APPLICABLE = False  # Set to True to try 8-bit loading for the 7B model
LOAD_IN_8BIT = True  # If USE_QUANTIZATION_IF_APPLICABLE is True, this will be used
# LOAD_IN_4BIT = False # 4-bit is more aggressive, might have issues on MPS

print(f"Attempting to load model: {CURRENT_MODEL_ID}")
if USE_QUANTIZATION_IF_APPLICABLE:
    if LOAD_IN_8BIT:
        print("Attempting to load in 8-bit.")
    # elif LOAD_IN_4BIT:
    #     print("Attempting to load in 4-bit.")

# Device configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
    # For CUDA, float16 is standard for good performance and memory balance
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    # For MPS, float32 is generally more stable, especially if not quantizing.
    # If quantizing, the model's internal compute might still effectively be lower precision.
    DTYPE = torch.float32
else:
    DEVICE = "cpu"
    DTYPE = torch.float32  # CPU usually uses float32

print(f"Using device: {DEVICE}")
print(f"Using base dtype for loading: {DTYPE}")

# --- Example Data ---
IMAGE_PATHS = []
if not IMAGE_PATHS:
    print("Warning: IMAGE_PATHS is empty. Creating dummy images for script testing.")
    try:
        os.makedirs("dummy_images", exist_ok=True)
        for i in range(3):  # Create 3 dummy images
            dummy_img_path = f"dummy_images/dummy_image_{i + 1}.png"
            if not os.path.exists(dummy_img_path):
                colors = ['red', 'green', 'blue']
                Image.new('RGB', (224, 224), color=colors[i % len(colors)]).save(dummy_img_path)
            IMAGE_PATHS.append(dummy_img_path)
        print(f"Using dummy images: {IMAGE_PATHS}")
    except Exception as e:
        print(f"Could not create dummy images: {e}. Please provide valid IMAGE_PATHS.")
        exit()

QUESTION_TEXT = "These images show a sequence of events. What is the correct temporal order?"
OPTIONS_TEXT = "A) Image 1, Image 2, Image 3. B) Image 2, Image 1, Image 3. C) Image 3, Image 1, Image 2."
FULL_QUESTION_WITH_OPTIONS = f"{QUESTION_TEXT}\nOptions: {OPTIONS_TEXT}\nPlease answer with the option letter only (e.g., A, B, C)."

# --- 2. Load Model and Processor ---
model_kwargs = {
    "torch_dtype": DTYPE,
    "low_cpu_mem_usage": True,
    "trust_remote_code": True
}

if USE_QUANTIZATION_IF_APPLICABLE and DEVICE != "cpu":  # Quantization typically for GPU/MPS
    if LOAD_IN_8BIT:
        print("Applying 8-bit quantization arguments.")
        model_kwargs["load_in_8bit"] = True
    # elif LOAD_IN_4BIT: # 4-bit can be added similarly if desired and supported
    #     print("Applying 4-bit quantization arguments.")
    #     model_kwargs["load_in_4bit"] = True
    # For quantized models, torch_dtype might be overridden or handled differently by bitsandbytes
    # Often, you might still specify torch.float16 for the compute type with 8-bit/4-bit weights.
    # If using quantization, bitsandbytes handles the dtype of the weights.
    # The DTYPE here would be for activations if not fully quantized.
    # model_kwargs["torch_dtype"] = torch.float16 # Often recommended with quantization

try:
    print(f"Loading model: {CURRENT_MODEL_ID} with kwargs: {model_kwargs}")
    model = LlavaForConditionalGeneration.from_pretrained(
        CURRENT_MODEL_ID,
        **model_kwargs
    )
    # If not quantizing, or if quantization doesn't move to device automatically:
    if not (USE_QUANTIZATION_IF_APPLICABLE and (LOAD_IN_8BIT)):  # or LOAD_IN_4BIT
        model = model.to(DEVICE)

    model.eval()
    print(
        f"Model loaded successfully. Model is on: {next(model.parameters()).device}, Dtype: {next(model.parameters()).dtype}")

    processor = AutoProcessor.from_pretrained(CURRENT_MODEL_ID, trust_remote_code=True)
    print("Processor loaded successfully.")

except Exception as e:
    print(f"Error loading model or processor for '{CURRENT_MODEL_ID}': {e}")
    print("Please ensure:")
    print(
        "1. You have followed the LLaVA-NeXT-Interleave setup instructions from their official repository (clone, install requirements).")
    print("2. All specific dependencies (like `bitsandbytes` if quantizing) are installed.")
    print("3. Model weights are accessible.")
    print("4. `trust_remote_code=True` is set.")
    print(
        "5. If on MPS and still OOM, try the PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 environment variable as a last resort before running.")
    exit()

# --- 3. Prepare Inputs for Multi-Image Inference ---
loaded_images = []
for image_path in IMAGE_PATHS:
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'. Please check paths.")
        exit()
    try:
        image = Image.open(image_path).convert("RGB")
        loaded_images.append(image)
    except Exception as e:
        print(f"Error opening or converting image '{image_path}': {e}")
        exit()

if not loaded_images:
    print("Error: No images were loaded. Please provide valid image paths.")
    exit()

content_parts = []
for _ in loaded_images:
    content_parts.append({"type": "image"})
content_parts.append({"type": "text", "text": FULL_QUESTION_WITH_OPTIONS})

messages = [{"role": "user", "content": content_parts}]

try:
    prompt_text_for_processor = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"\nFormatted prompt text for processor (using chat template):\n{prompt_text_for_processor}")

    # For inputs, it's generally good to match the model's compute dtype if possible,
    # especially if not using quantization that changes dtypes internally.
    # The processor should handle image preprocessing.
    # The .to(DEVICE) call for inputs should happen after processing.
    inputs = processor(
        text=prompt_text_for_processor,
        images=loaded_images,
        return_tensors="pt"
    ).to(DEVICE)  # Move all processed inputs to the target device

    # Ensure input tensor dtypes are compatible with the model if not automatically handled
    # For example, if model is float16, inputs['pixel_values'] might need to be float16
    if 'pixel_values' in inputs and inputs['pixel_values'].dtype != model.dtype:
        print(f"Converting pixel_values dtype from {inputs['pixel_values'].dtype} to {model.dtype}")
        inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)

    print("\nInputs successfully processed for the model.")
    # print(f"Input shapes: input_ids: {inputs['input_ids'].shape}, pixel_values: {inputs['pixel_values'].shape if 'pixel_values' in inputs else 'N/A'}")
    # print(f"Input dtypes: input_ids: {inputs['input_ids'].dtype}, pixel_values: {inputs['pixel_values'].dtype if 'pixel_values' in inputs else 'N/A'}")


except Exception as e:
    print(f"Error during input processing with LLaVA processor: {e}")
    print("Critical: Verify the multi-image input format and chat template usage for your LLaVA-Interleave model.")
    exit()

# --- 4. Generate Response ---
try:
    print("\nGenerating response from LLaVA...")
    generation_kwargs = {
        "max_new_tokens": 50,
        "num_beams": 1,
        "do_sample": False,
    }

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    if 'input_ids' in inputs:
        input_token_len = inputs['input_ids'].shape[1]
        response_ids = generated_ids[:, input_token_len:]
    else:
        response_ids = generated_ids
        print(
            "Warning: Decoding full generated sequence as input_ids not found in processor output. Response may include prompt.")

    generated_text = processor.batch_decode(response_ids, skip_special_tokens=True)[0]

    print("\n--- LLaVA Model Raw Response ---")
    print(generated_text.strip())
    print("------------------------------")

except Exception as e:
    print(f"Error during model generation: {e}")
    print("Check for out-of-memory errors or issues with generation arguments.")

print("\nScript finished.")
