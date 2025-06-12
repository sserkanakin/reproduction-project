import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig
from PIL import Image


# --- 1. Argument Parsing ---
def parse_args():
    """Parses command-line arguments for the fine-tuning script."""
    parser = argparse.ArgumentParser(description="A robust LoRA Fine-tuning script for LLaVA models.")

    # Model and Quantization
    parser.add_argument("--base_model_id", type=str, default="llava-hf/llava-interleave-qwen-7b-hf")
    parser.add_argument("--use_quantization", action="store_true", help="Enable 8-bit quantization.")

    # Data
    parser.add_argument("--finetuning_dataset_path", type=str, required=True)
    parser.add_argument("--image_base_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Training
    parser.add_argument("--output_dir", type=str, default="./llava_finetuned_adapters")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)

    return parser.parse_args()


# --- 2. Custom Dataset ---
class FineTuningDataset(Dataset):
    """A simple dataset class that loads text and images."""

    def __init__(self, dataset_path, image_base_dir):
        self.image_base_dir = image_base_dir
        with open(dataset_path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load images
        images = []
        image_paths = item.get("source_images", [])
        if image_paths:
            for path in image_paths:
                try:
                    full_path = os.path.join(self.image_base_dir, path.lstrip('./'))
                    images.append(Image.open(full_path).convert("RGB"))
                except Exception as e:
                    print(f"Warning: Could not load image {full_path}. Skipping sample. Error: {e}")
                    return None  # The collator will filter this out

        return {
            "instruction": item["instruction"],
            "output": item["output"],
            "images": images
        }


# --- 3. Custom Data Collator ---
class DataCollatorForLLaVAFinetuning:
    """
    A custom data collator that handles tokenization, creating labels,
    and preparing the final model inputs.
    """

    def __init__(self, processor, max_length):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features):
        # Filter out samples that failed to load
        features = [f for f in features if f is not None]
        if not features:
            return {}

        # Combine instruction and output for each sample
        full_texts = [f["instruction"] + f["output"] for f in features]
        instruction_texts = [f["instruction"] for f in features]

        # Process images. We need to flatten the list of lists of images.
        images = [img for f in features for img in f["images"]]

        # Assign each image to its corresponding sample in the batch
        image_batch_indices = []
        for i, f in enumerate(features):
            image_batch_indices.extend([i] * len(f["images"]))

        # Tokenize the full texts and process images
        inputs = self.processor(
            text=full_texts,
            images=images,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_batch_segments_labels=True,  # For Llava-Interleave
            image_batch_indices=image_batch_indices  # For Llava-Interleave
        )

        # Create labels by cloning input_ids
        labels = inputs.input_ids.clone()

        # Tokenize instructions separately to find their lengths for masking
        instruction_token_lengths = [
            len(self.processor.tokenizer(inst, add_special_tokens=False).input_ids)
            for inst in instruction_texts
        ]

        # Mask the instruction part of the labels
        # The processor adds a BOS token, so we account for it with `+1`
        for i, length in enumerate(instruction_token_lengths):
            labels[i, :length + 1] = -100

        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


# --- 4. Custom Trainer ---
class MultiImageLlavaTrainer(Trainer):
    """Custom Trainer to handle potential 5D pixel_values tensor."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "pixel_values" in inputs and inputs["pixel_values"] is not None and inputs["pixel_values"].ndim == 5:
            bs, num_images, c, h, w = inputs["pixel_values"].shape
            inputs["pixel_values"] = inputs["pixel_values"].view(bs * num_images, c, h, w)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


# --- 5. Main Execution Block ---
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("--- Starting LLaVA LoRA Fine-tuning ---")

    # Load model and processor
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.use_quantization else None

    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)

    # Configure tokenizer
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Setup LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset and data collator
    train_dataset = FineTuningDataset(args.finetuning_dataset_path, args.image_base_dir)
    data_collator = DataCollatorForLLaVAFinetuning(processor, args.max_seq_length)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        fp16=True,
        optim="paged_adamw_8bit",
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=False,  # We need to keep original columns for the collator
        report_to="none",
    )

    # Initialize Trainer
    trainer = MultiImageLlavaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("Starting fine-tuning process...")
    trainer.train()

    print("\n--- Fine-tuning Complete ---")
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_checkpoint_dir)
    print(f"Final LoRA adapters saved to: {final_checkpoint_dir}")


if __name__ == "__main__":
    main()