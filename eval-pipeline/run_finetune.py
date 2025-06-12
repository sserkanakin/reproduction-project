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
    # Handle relative paths and base data directory
    p = Path(img_path)
    # Try as given
    if p.is_absolute() and p.exists():
        return str(p)
    rel = Path(os.getcwd()) / p
    if rel.exists():
        return str(rel)
    # Try under eval-pipeline/data
    alt = Path(os.getcwd()) / 'eval-pipeline' / 'data' / p
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(f"Image file not found: {img_path}")


def preprocess(example, processor, max_in=512, max_out=256, max_images=6):
    # example fields: source_images, instruction, output
    # 1. Process images independently
    image_tensors = []
    for img_path in example['source_images'][:max_images]:
        full_path = resolve_image_path(img_path)
        img = Image.open(full_path).convert('RGB')
        pixel = processor.image_processor(images=img, return_tensors='pt').pixel_values
        image_tensors.append(pixel)
    # pad/truncate to max_images
    if len(image_tensors) < max_images:
        pad = torch.zeros_like(image_tensors[0])
        image_tensors += [pad] * (max_images - len(image_tensors))
    pixel_values = torch.cat(image_tensors, dim=1).squeeze(0)

    # 2. Tokenize instruction text separately
    tokenized = processor.tokenizer(
        example['instruction'],
        truncation=True,
        padding='max_length',
        max_length=max_in,
        return_tensors='pt'
    )
    input_ids = tokenized.input_ids.squeeze(0)
    attention_mask = tokenized.attention_mask.squeeze(0)

    # 3. Tokenize output (reasoning + final answer) for labels
    labels = processor.tokenizer(
        example['output'],
        truncation=True,
        padding='max_length',
        max_length=max_out,
        return_tensors='pt'
    ).input_ids.squeeze(0)

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
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
        device_map='auto',
        torch_dtype=torch.float16 if args.fp16 else None,
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
        save_total_limit=2,
        logging_steps=50,
        do_eval=True,
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
