import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA with LoRA on a vision-text sequence-to-sequence task"
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
        help="Path to JSONL file containing training examples"
    )
    parser.add_argument(
        "--val_file", type=str, required=True,
        help="Path to JSONL file containing validation examples"
    )
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="Pretrained LLaVA model identifier (e.g., llava-hf/llava-interleave-qwen-7b-hf)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="lora-output",
        help="Directory to save LoRA-adapted model and checkpoints"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=8,
        help="LoRA rank (bottleneck dimension)"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable 16-bit mixed precision training"
    )
    parser.add_argument(
        "--in_8bit", action="store_true",
        help="Load model in 8-bit via bitsandbytes for memory efficiency"
    )
    return parser.parse_args()


def resolve_image_path(img_path: str) -> str:
    # Handle relative paths by resolving against the current working directory (/app)
    p = Path(img_path)
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    return str(p)


def preprocess(example, processor, max_in=512, max_out=256):
    # example fields: source_images, instruction, output
    image_tensors = []
    for img_path in example['source_images']:
        full_path = resolve_image_path(img_path)
        img = Image.open(full_path).convert('RGB')
        pixel = processor(images=img, return_tensors='pt').pixel_values
        image_tensors.append(pixel)
    # concatenate pixels along width dimension
    pixel_values = torch.cat(image_tensors, dim=1).squeeze(0)

    # Tokenize instruction text
    tokenized_inputs = processor.tokenizer(
        example['instruction'],
        truncation=True,
        padding='max_length',
        max_length=max_in,
        return_tensors='pt',
    )
    # Tokenize output (reasoning + final answer)
    tokenized_labels = processor.tokenizer(
        example['output'],
        truncation=True,
        padding='max_length',
        max_length=max_out,
        return_tensors='pt',
    )

    return {
        'pixel_values': pixel_values,
        'input_ids': tokenized_inputs.input_ids.squeeze(0),
        'attention_mask': tokenized_inputs.attention_mask.squeeze(0),
        'labels': tokenized_labels.input_ids.squeeze(0),
    }


def main():
    args = parse_args()

    # Load dataset from JSONL
    data_files = {'train': args.train_file, 'validation': args.val_file}
    ds = load_dataset('json', data_files=data_files)

    # Load processor & base model
    print(f"Loading model {args.base_model} (8-bit={args.in_8bit})...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        load_in_8bit=args.in_8bit,
        device_map='auto'
    )

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=args.lora_dropout,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # Preprocess examples (images + text)
    print("Tokenizing and processing images...")
    ds = ds.map(
        lambda ex: preprocess(ex, processor),
        remove_columns=['id', 'source_images', 'instruction', 'output', 'ground_truth_option'],
        batched=False
    )

    # Set up training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        evaluation_strategy='epoch',
        save_total_limit=2,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        data_collator=default_data_collator,
    )

    # Train and save
    trainer.train()
    print(f"Saving LoRA model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
