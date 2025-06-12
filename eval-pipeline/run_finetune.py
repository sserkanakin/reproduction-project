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
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig
from PIL import Image


def parse_args():
    """Parses command-line arguments for the fine-tuning script."""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning script for LLaVA models.")
    parser.add_argument("--base_model_id", type=str, default="llava-hf/llava-interleave-qwen-7b-hf")
    parser.add_argument("--use_quantization", action="store_true", help="Enable 8-bit quantization for VRAM saving.")
    parser.add_argument("--finetuning_dataset_path", type=str, required=True,
                        help="Path to the .jsonl fine-tuning data.")
    parser.add_argument("--image_base_dir", type=str, required=True,
                        help="Base directory where MMIU images are stored.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for truncation.")
    parser.add_argument("--output_dir", type=str, default="./llava_finetuned_adapters",
                        help="Directory to save adapters.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save a checkpoint every X steps.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps to run (overrides epochs).")
    return parser.parse_args()


class FineTuningDataset(Dataset):
    """Simple Dataset to load and process data for fine-tuning."""

    def __init__(self, dataset_path, processor, image_base_dir, max_length):
        self.processor = processor
        self.image_base_dir = image_base_dir
        self.max_length = max_length
        with open(dataset_path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        full_text = item['instruction'] + item['output']

        images = []
        if image_paths := item.get("source_images"):
            for path in image_paths:
                try:
                    full_path = os.path.join(self.image_base_dir, path.lstrip('./'))
                    images.append(Image.open(full_path).convert("RGB"))
                except Exception:
                    # Return None if an image fails to load; collator will skip it.
                    return None

        try:
            # The processor handles tokenization, image processing, and truncation.
            inputs = self.processor(
                text=full_text,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            # For Causal LM, the labels are the same as the input_ids.
            # The model internally handles masking the prompt part during loss calculation.
            inputs['labels'] = inputs['input_ids'].clone()

            # Remove the extra batch dimension.
            return {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception:
            # If any other processing error occurs, skip the sample.
            return None


def custom_data_collator(features):
    """A simple data collator that filters out failed samples and batches them."""
    # Filter out None values which may have been returned by the dataset's __getitem__
    features = [f for f in features if f is not None]
    if not features:
        return {}
    # Batch the features
    return {key: torch.stack([f[key] for f in features]) for key in features[0].keys()}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("--- Starting LLaVA LoRA Fine-tuning ---")

    # Setup quantization
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if args.use_quantization else None

    # Load model and processor
    model = LlavaForConditionalGeneration.from_pretrained(args.base_model_id, torch_dtype=torch.float16,
                                                          quantization_config=quant_config, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)

    # Configure tokenizer padding
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset instance
    train_dataset = FineTuningDataset(args.finetuning_dataset_path, processor, args.image_base_dir, args.max_seq_length)

    # Define training arguments
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
        remove_unused_columns=False,
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_data_collator
    )

    # Start fine-tuning
    print("Starting the training process...")
    trainer.train()

    # Save the final model
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_checkpoint_dir)
    print(f"\n--- Fine-tuning Complete --- \nFinal LoRA adapters saved to: {final_checkpoint_dir}")


if __name__ == "__main__":
    main()