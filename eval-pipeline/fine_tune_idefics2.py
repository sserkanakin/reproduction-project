# fine_tune_idefics2_v11_definitive.py

"""
A professional script to fine-tune the Idefics2-8b model.

VERSION 11 (DEFINITIVE FIX 2):

- Corrected the logic inside the `CustomDataCollator` to robustly handle prompt creation, permanently fixing the recurring `ValueError: shape mismatch`.
- The collator now correctly parses the human prompt, splitting it by the <image> placeholder and interleaving the text chunks with the actual image objects.
- The Dataset is now extremely simple, only loading raw data. All complex processing is now correctly handled within the collator.

How to Run:

python fine_tune_idefics2_v11_definitive.py \
  --train_file ./data/finetune_data/train.json \
  --val_file ./data/finetune_data/eval.json \
  --image_base_path ./data/finetune_data/ \
  --output_dir ./idefics2-8b-temporal-finetune-a100 \
  --epochs 1
"""

import os
import argparse
import logging
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from PIL import Image
import bitsandbytes as bnb

from transformers import (
    AutoProcessor,
    Idefics2ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- 1. Definitive Custom Data Collator ---
class CustomDataCollator:
    def __init__(self, processor: AutoProcessor, image_base_path: str, max_length: int = 2048):
        self.processor = processor
        self.image_base_path = image_base_path
        self.image_token_id = -200  # Idefics2 uses -200 for image tokens in labels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images_batch = []

        for example in examples:
            try:
                image_paths = [os.path.join(self.image_base_path, path.lstrip('./')) for path in example["image"]]
                images = [Image.open(path).convert("RGB") for path in image_paths]
            except FileNotFoundError as e:
                logger.error(f"Skipping sample due to missing image: {e}")
                continue

            human_text = example["conversations"][0]["value"]
            gpt_text = example["conversations"][1]["value"]

            # --- ROBUST INTERLEAVING LOGIC ---
            text_chunks = human_text.split("<image>")
            content = []
            if text_chunks[0]:
                content.append({"type": "text", "text": text_chunks[0]})
            for i, image in enumerate(images):
                content.append({"type": "image"})
                if (i + 1) < len(text_chunks) and text_chunks[i + 1]:
                    content.append({"type": "text", "text": text_chunks[i + 1]})

            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": gpt_text},
            ]

            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images_batch.append(images)

        batch = self.processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            padding=True
        )

        labels = batch["input_ids"].clone()
        # Important: Use -100 to ignore loss on padding and image tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        return batch

def main(args):
    logger.info(f"Loading base model: {args.model_id}")

    model = Idefics2ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    logger.info("Model and processor loaded.")

    logger.info(f"Configuring LoRA with rank r={args.lora_r}...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["connector", "lm_head"],
    )

    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    model.print_trainable_parameters()
    logger.info("LoRA configured.")

    data_collator = CustomDataCollator(processor, args.image_base_path)

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    val_dataset = load_dataset("json", data_files=args.val_file, split="train")

    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if args.paged_optimizer else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting fine-tuning with the definitive data collator...")
    trainer.train()
    logger.info("Training complete.")

    final_save_path = os.path.join(args.output_dir, "final_checkpoint")
    logger.info(f"Saving the final fine-tuned model adapter to {final_save_path}")
    model.save_pretrained(final_save_path)
    processor.save_pretrained(final_save_path)
    logger.info("Script finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the Idefics2-8b model using a robust data collator.")
    parser.add_argument("--model_id", type=str, default="HuggingFaceM4/idefics2-8b", help="Hugging Face model ID.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training JSON file.")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation JSON file.")
    parser.add_argument("--image_base_path", type=str, required=True, help="Base directory where image folders are stored.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha scaling parameter.")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=True, help="Enable gradient checkpointing.")
    parser.add_argument("--paged_optimizer", action='store_true', default=True, help="Use paged AdamW optimizer.")
    args = parser.parse_args()
    main(args)