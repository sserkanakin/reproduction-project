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


# --- 1. Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Robust LoRA Fine-tuning script for LLaVA models.")
    parser.add_argument("--base_model_id", type=str, default="llava-hf/llava-interleave-qwen-7b-hf")
    parser.add_argument("--use_quantization", action="store_true")
    parser.add_argument("--finetuning_dataset_path", type=str, required=True)
    parser.add_argument("--image_base_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./llava_finetuned_adapters")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)
    return parser.parse_args()


# --- 2. Simplified Dataset Class ---
class LlavaFinetuningDataset(Dataset):
    """
    A simple dataset class that loads raw data: text prompts and images.
    The heavy processing is deferred to the data collator.
    """

    def __init__(self, dataset_path: str, image_base_dir: str):
        self.image_base_dir = image_base_dir
        with open(dataset_path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, List]:
        item = self.dataset[idx]
        images = []
        if image_paths := item.get("source_images"):
            for path in image_paths:
                try:
                    full_path = os.path.join(self.image_base_dir, path.lstrip('./'))
                    images.append(Image.open(full_path).convert("RGB"))
                except Exception as e:
                    print(f"Warning: Could not load image {full_path}. Skipping sample. Error: {e}")
                    return None  # This sample will be filtered out by the collator.

        return {
            "instruction": item["instruction"],
            "output": item["output"],
            "images": images
        }


# --- 3. Robust Data Collator Class ---
# python
class DataCollatorForLlavaFinetuning:
    def __init__(self, processor: AutoProcessor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if f is not None]
        if not features:
            return {}

        all_images = [img for f in features for img in f["images"]]
        prompts = [f["instruction"] + f["output"] for f in features]

        inputs = self.processor(
            text=prompts,
            images=all_images,
            padding="max_length",
            truncation=False,  # Disabled truncation to avoid image token count mismatch
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = inputs.input_ids.clone()

        instruction_prompts = [f["instruction"] for f in features]
        instruction_token_lengths = [
            len(self.processor.tokenizer(inst, add_special_tokens=False).input_ids)
            for inst in instruction_prompts
        ]

        for i, length in enumerate(instruction_token_lengths):
            labels[i, :length + 1] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


# --- 4. Main Execution Block ---
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Starting LLaVA LoRA Fine-tuning ---")

    # Model and Quantization Setup
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if args.use_quantization else None
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_id, torch_dtype=torch.float16,
        quantization_config=quant_config, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)

    # Tokenizer Setup
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    # LoRA Setup
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset and Data Collator Setup
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
        remove_unused_columns=False, report_to="none"
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

    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_checkpoint_dir)
    print(f"\n--- Fine-tuning Complete --- \nFinal LoRA adapters saved to: {final_checkpoint_dir}")


if __name__ == "__main__":
    main()