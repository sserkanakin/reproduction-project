import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
)
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning script for a LLaVA model.")
    # --- Model Arguments ---
    parser.add_argument("--base_model_id", type=str, default="llava-hf/llava-interleave-qwen-7b-hf",
                        help="The Hugging Face Hub ID of the base LLaVA model to fine-tune.")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Enable 8-bit quantization for the base model to save VRAM during training.")

    # --- Data Arguments ---
    parser.add_argument("--finetuning_dataset_path", type=str, required=True,
                        help="Path to the .jsonl file containing the fine-tuning data.")
    parser.add_argument("--image_base_dir", type=str, required=True,
                        help="Base directory where MMIU images are stored.")

    # --- Training Arguments ---
    parser.add_argument("--output_dir", type=str, default="./llava_finetuned_adapters",
                        help="Directory to save the fine-tuned LoRA adapters and training checkpoints.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device during training. Increase if VRAM allows.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients before performing an optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for the optimizer.")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log training state every X steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save a checkpoint every X steps.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If set to a positive number, overrides num_train_epochs and runs for this many steps. Ideal for testing.")

    return parser.parse_args()


class LlavaFinetuneDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for LLaVA fine-tuning."""

    def __init__(self, dataset_path, processor, image_base_dir):
        self.processor = processor
        self.image_base_dir = image_base_dir
        with open(dataset_path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        full_text = item['instruction'] + item['output']

        images = []
        image_paths = item.get("source_images", [])
        if image_paths:
            for path in image_paths:
                try:
                    full_path = os.path.join(self.image_base_dir, path)
                    if path.startswith('./'):
                        full_path = os.path.join(self.image_base_dir, path[2:])
                    image = Image.open(full_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(
                        f"Warning: Could not load image {full_path} for sample {item.get('id', 'unknown')}. Skipping. Error: {e}")
                    return None

        try:
            # The processor prepares the combined text-image input.
            # Padding is handled by the data collator in the Trainer.
            inputs = self.processor(text=full_text, images=images if images else None, return_tensors="pt")
            # Squeeze to remove the batch dimension that the processor adds by default.
            # The data collator will add it back for the batch.
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            return inputs
        except Exception as e:
            print(f"Error processing sample {item.get('id', 'unknown')}. Error: {e}")
            return None


def custom_data_collator(features):
    """Filter out None items from a batch and let default collator handle padding."""
    # Filter out samples that failed to load
    features = [f for f in features if f is not None]
    if not features:
        return {}

    # Let the default collator from transformers handle the rest.
    # It's smart enough to pad correctly if the processor's pad_token is set.
    from transformers.data.data_collator import default_data_collator
    return default_data_collator(features)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Starting LoRA Fine-tuning...")

    # --- Load Model and Processor ---
    quantization_config = None
    if args.use_quantization:
        print("Using 8-bit quantization.")
        # Use BitsAndBytesConfig for modern transformers versions
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto"  # Let accelerate handle device placement
    )
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)

    # --- CORRECTED: Set padding token on the tokenizer, not the processor ---
    # The tokenizer is a component of the processor object
    if processor.tokenizer.pad_token is None:
        print("pad_token not set. Setting it to eos_token for padding.")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        # Also update the model's config to use this pad_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Set padding side to the right for training causal language models
    processor.tokenizer.padding_side = "right"

    # --- Setup LoRA / PEFT ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("Applying LoRA adapters to the model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load and Prepare Dataset ---
    print(f"Loading and processing dataset from: {args.finetuning_dataset_path}")
    train_dataset = LlavaFinetuneDataset(
        dataset_path=args.finetuning_dataset_path,
        processor=processor,
        image_base_dir=args.image_base_dir
    )

    # --- Setup Trainer ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        report_to="none",
        fp16=True,  # Use mixed precision training
        remove_unused_columns=False,  # Important for custom datasets
        save_total_limit=2,  # Optional: only keep the last 2 checkpoints
        dataloader_num_workers=2,  # Optional: for faster data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Fine-tuning complete. Saving final LoRA adapters.")
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_checkpoint_dir)
    print(f"Final adapters saved to: {final_checkpoint_dir}")


if __name__ == "__main__":
    main()
