import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from peft import LoraConfig, get_peft_model

class MultiImageDataset(Dataset):
    def __init__(self, jsonl_file, image_dir, processor, max_length=2048):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        with open(jsonl_file, 'r') as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # load & verify all images
        imgs = []
        for rel in ex["source_images"]:
            path = os.path.join(self.image_dir, rel)
            if not os.path.exists(path):
                return None
            img = Image.open(path).convert("RGB")
            imgs.append(img)

        # tokenize image+instruction
        enc = self.processor(
            images=imgs,
            text=ex["instruction"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to_dict()

        # tokenize the assistant’s output as labels
        labels = self.processor.tokenizer(
            ex["output"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).input_ids

        enc["labels"] = torch.tensor(labels)
        # squeeze out the batch dim
        return {k: v.squeeze(0) for k, v in enc.items()}

def collate_fn(batch):
    # drop any None’s
    batch = [b for b in batch if b is not None]
    return default_data_collator(batch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",     type=str, required=True)
    parser.add_argument("--dataset_path",   type=str, required=True)
    parser.add_argument("--image_base_dir", type=str, required=True)
    parser.add_argument("--output_dir",     type=str, required=True)
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--accum_steps",    type=int, default=8)
    parser.add_argument("--max_steps",      type=int, default=20)
    args = parser.parse_args()

    # processor & model
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        quantization_config=None  # or your bitsandbytes config
    ).cuda()

    # attach LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)

    # dataset & trainer
    train_ds = MultiImageDataset(
        jsonl_file=args.dataset_path,
        image_dir=args.image_base_dir,
        processor=processor
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        max_steps=args.max_steps,
        fp16=True,
        save_steps=10,
        logging_steps=5,
        learning_rate=5e-5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_pretrained(args.output_dir)
