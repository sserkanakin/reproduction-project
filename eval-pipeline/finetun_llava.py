import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from PIL import Image
import os

# --- 1. Configuration ---
# The model we are fine-tuning
model_id = "llava-hf/llava-interleave-qwen-7b-hf"

# Path to your JSON dataset file
dataset_path = "/data/finetuning_data/test/llava_temporal_train.jsonl"  # <--- IMPORTANT: SET THIS TO YOUR FILENAME

# Directory to save the fine-tuned model adapter
output_dir = "./llava-finetuned-temporal-adapter"

# --- 2. Load Model and Processor with 4-bit Quantization (Q-LoRA) ---
# This is the key to fitting the model into L4 GPU memory
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading base model and processor...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically maps model layers to available devices
)

processor = AutoProcessor.from_pretrained(model_id)

# --- 3. Configure Parameter-Efficient Fine-Tuning (PEFT) with LoRA ---
# We only train a small set of "adapter" weights, not the whole model
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices. Higher rank means more parameters to train.
    lora_alpha=32,  # Alpha parameter for scaling.
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target layers for LoRA
    bias="none",
    task_type="CAUSAL_LM",
)

# Add LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 4. Prepare the Dataset and Formatting ---
# Load your dataset from the JSON file
raw_dataset = load_dataset("json", data_files=dataset_path, split="train")


def formatting_prompts_func(example):
    """
    This function formats a single data point into the prompt string
    that the model expects. It concatenates the user and assistant turns.
    """
    # The LLaVA prompt template includes system message, user, and assistant roles
    # The <image> tokens are placeholders that the processor will handle.

    user_prompt = ""
    assistant_response = ""

    # Extract user and assistant content from the 'conversations' list
    for turn in example['conversations']:
        if turn['role'] == 'user':
            user_prompt = turn['content']
        elif turn['role'] == 'assistant':
            assistant_response = turn['content']

    # We must provide the full conversation string
    full_prompt = f"USER: {user_prompt}\nASSISTANT: {assistant_response}"
    return [full_prompt]


# The SFTTrainer requires a custom data collator for multi-modal models
# This collator will handle both text and image processing
class LlavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Each 'feature' is a dictionary from our dataset
        # We need to separate images from the text prompts
        prompts = [f['text'] for f in features]
        image_lists = [f['images'] for f in features]

        # Load all images. Note that we have a list of lists of image paths.
        images = []
        for paths in image_lists:
            # For each data point, load all its images
            loaded_images = [Image.open(p).convert("RGB") for p in paths]
            images.append(loaded_images)

        # Process the batch
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        return inputs


# Pre-process the dataset to add a 'text' column using our formatting function
# This is needed for the SFTTrainer
def preprocess_dataset(example):
    example['text'] = formatting_prompts_func(example)[0]
    return example


processed_dataset = raw_dataset.map(preprocess_dataset,
                                    remove_columns=next(iter(raw_dataset))['conversations'][0].keys())

# --- 5. Configure Training Arguments ---
# These arguments are optimized for an L4 GPU
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,  # A single epoch is often enough for fine-tuning
    per_device_train_batch_size=1,  # Batch size of 1 is crucial for L4 memory
    gradient_accumulation_steps=8,  # Simulate a larger batch size (1 * 8 = 8)
    learning_rate=1e-4,
    fp16=True,  # Use mixed-precision training
    logging_steps=10,
    save_total_limit=2,
    report_to="none",  # Disable reporting to services like WandB
)

# --- 6. Initialize and Run the Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=lora_config,
    args=training_args,
    formatting_func=lambda example: formatting_prompts_func(example),
    data_collator=LlavaDataCollator(processor),
)

print("Starting the fine-tuning process...")
trainer.train()

# --- 7. Save the Final Model ---
print(f"Training complete. Saving model adapter to {output_dir}")
trainer.save_model(output_dir)