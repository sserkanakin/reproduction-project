#!/usr/bin/env python
import os
import json
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import (
    LlavaProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
)
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

from peft import LoraConfig, get_peft_model

class MultiImageDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, processor, max_length=2048):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        with open(jsonl_path, 'r') as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # 1) load images
        imgs = []
        for rel_path in ex["source_images"]:
            full_path = os.path.join(self.image_dir, rel_path)
            if not os.path.exists(full_path):
                # drop this example if an image is missing
                return None
            img = Image.open(full_path).convert("RGB")
            imgs.append(img)

        # 2) encode vision + instruction
        encodings = self.processor(
            images=imgs,
            text=ex["instruction"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # 3) encode assistant output as labels
        labels = self.processor.tokenizer(
            ex["output"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).input_ids

        # squeeze off the batch dim
        batch = {k: v.squeeze(0) for k, v in encodings.to_dict().items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

def collate_fn(batch):
    # drop None examples
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    return default_data_collator(batch)

def main():
    parser = argparse.ArgumentParser(description="Finetune LLaVA with PEFT")
    parser.add_argument("--base_model_id",     type=str,  required=True,
                        help="e.g. llava-hf/llava-interleave-qwen-0.5b-hf")
    parser.add_argument("--finetuning_dataset_path", type=str, required=True)
    parser.add_argument("--image_base_dir",    type=str,  required=True)
    parser.add_argument("--output_dir",        type=str,  required=True)
    parser.add_argument("--use_quantization",  action="store_true")
    parser.add_argument("--max_seq_length",    type=int,  default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps",         type=int,  default=20)
    parser.add_argument("--learning_rate",     type=float, default=5e-5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # — processor —
    processor = LlavaProcessor.from_pretrained(
        args.base_model_id,
        use_fast=True
    )

    # — quantization config if needed —
    bnb_config = None
    if args.use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    # — base model —
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config
    )

    # — attach LoRA adapters —
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # — dataset & trainer setup —
    train_dataset = MultiImageDataset(
        jsonl_path=args.finetuning_dataset_path,
        image_dir=args.image_base_dir,
        processor=processor,
        max_length=args.max_seq_length
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=5,
        save_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # — run training & save —
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
