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
)
from peft import get_peft_model, LoraConfig
from PIL import Image
from typing import Dict, List


def parse_args():
    """Parses command-line arguments for the fine-tuning script."""
    parser = argparse.ArgumentParser(description="Robust LoRA Fine-tuning script for LLaVA models on high-memory GPUs.")

    # Model and Quantization Arguments
    parser.add_argument("--base_model_id", type=str, default="llava-hf/llava-interleave-qwen-7b-hf",
                        help="Base model ID from Hugging Face Hub.")
    parser.add_argument("--use_quantization", action="store_true", help="Enable 8-bit quantization for VRAM saving.")

    # Data Arguments
    parser.add_argument("--finetuning_dataset_path", type=str, required=True,
                        help="Path to the .jsonl fine-tuning data.")
    parser.add_argument("--image_base_dir", type=str, required=True,
                        help="Base directory where MMIU images are stored.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length for tokenization.")

    # Training Arguments
    parser.add_argument("--output_dir", type=str, default="./llava_finetuned_adapters",
                        help="Directory to save adapters.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device. Increase if VRAM allows.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save a checkpoint every X steps.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps to run (overrides epochs).")

    return parser.parse_args()


class LlavaFinetuningDataset(Dataset):
    """
    A simple dataset class that loads raw data: text prompts and images.
    The heavy processing is deferred to the data collator for robustness.
    """

    def __init__(self, dataset_path: str, image_base_dir: str):
        self.image_base_dir = image_base_dir
        with open(dataset_path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, any]:
        item = self.dataset[idx]
        images = []
        if image_paths := item.get("source_images"):
            for path in image_paths:
                try:
                    full_path = os.path.join(self.image_base_dir, path.lstrip('./'))
                    images.append(Image.open(full_path).convert("RGB"))
                except Exception as e:
                    print(f"Warning: Could not load image {full_path}. Error: {e}. Skipping sample.")
                    return None  # This sample will be filtered by the collator.

        return {
            "instruction": item["instruction"],
            "output": item["output"],
            "images": images
        }


class DataCollatorForLlavaFinetuning:
    """
    A robust data collator that handles all preprocessing, tokenization,
    and label creation for a batch of data. This is the standard, reliable
    approach for complex models like LLaVA.
    """

    def __init__(self, processor: AutoProcessor, max_length: int):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Filter out samples that failed to load (e.g., due to missing images).
        features = [f for f in features if f is not None]
        if not features:
            return {}

        # The LLaVA processor expects a flat list of all images in the batch.
        all_images = [img for f in features for img in f["images"]]

        # We need to manually control the text creation to avoid truncation errors.
        # This collator intelligently truncates the lengthy 'output' part while preserving
        # the critical 'instruction' part with all its <image> tokens.
        final_prompts = []
        instruction_lengths = []
        for f in features:
            # Tokenize instruction and output separately to manage their lengths.
            instruction_tokens = self.tokenizer(f["instruction"], add_special_tokens=False).input_ids
            output_tokens = self.tokenizer(f["output"], add_special_tokens=False).input_ids

            # Calculate the max length available for the output, accounting for the BOS token.
            max_output_len = self.max_length - len(instruction_tokens) - 1

            # Truncate the output tokens if they are too long.
            if len(output_tokens) > max_output_len:
                output_tokens = output_tokens[:max_output_len]

            # Reconstruct the text from the tokens. This is safer than string slicing.
            truncated_output = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

            final_prompt = f["instruction"] + truncated_output
            final_prompts.append(final_prompt)

            # Store the length of the instruction part to mask labels correctly.
            # Add 1 to account for the BOS token that the processor adds.
            instruction_lengths.append(len(instruction_tokens) + 1)

        # Process the entire batch at once using the carefully constructed prompts.
        inputs = self.processor(
            text=final_prompts,
            images=all_images,
            padding="max_length",
            truncation=True,  # This is now safe.
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Create labels for language modeling by cloning input_ids.
        labels = inputs.input_ids.clone()

        # Mask the instruction part of the labels so the model doesn't learn to predict the prompt.
        for i, length in enumerate(instruction_lengths):
            labels[i, :length] = -100  # -100 is the standard ignore_index in PyTorch.

        # Also mask any padding tokens in the labels.
        labels[labels == self.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Starting LLaVA LoRA Fine-tuning on V100 ---")

    # Model and Quantization Setup
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if args.use_quantization else None
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        trust_remote_code=True,
        # Enable gradient checkpointing to further optimize VRAM usage.
        gradient_checkpointing=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)

    # Tokenizer Setup
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset and Data Collator Initialization
    train_dataset = LlavaFinetuningDataset(args.finetuning_dataset_path, args.image_base_dir)
    data_collator = DataCollatorForLlavaFinetuning(processor, args.max_seq_length)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, logging_steps=args.logging_steps,
        save_steps=args.save_steps, max_steps=args.max_steps, fp16=True,
        optim="paged_adamw_8bit", save_total_limit=2,
        remove_unused_columns=False,  # Important: Collator needs original columns.
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting the training process...")
    trainer.train()

    # Save final model
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_checkpoint_dir)
    print(f"\n--- Fine-tuning Complete --- \nFinal LoRA adapters saved to: {final_checkpoint_dir}")


if __name__ == "__main__":
    main()
