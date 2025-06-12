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
    p = Path(img_path)
    if p.is_absolute() and p.exists():
        return str(p)
    rel = Path(os.getcwd()) / p
    if rel.exists():
        return str(rel)
    alt = Path(os.getcwd()) / 'eval-pipeline' / 'data' / p
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(f"Image file not found: {img_path}")


def preprocess(example, processor, max_images=6, max_out=256):
    # Load and pad/truncate images
    images = []
    for img_path in example['source_images'][:max_images]:
        full_path = resolve_image_path(img_path)
        images.append(Image.open(full_path).convert('RGB'))
    if len(images) < max_images:
        blank = Image.new('RGB', images[0].size, (0, 0, 0))
        images += [blank] * (max_images - len(images))

    # Use processor to handle images + instruction without truncating text
    tokenizer = processor.tokenizer
    text = example['instruction']
    max_len = tokenizer.model_max_length
    proc_inputs = processor(
        images=images,
        text=text,
        padding='max_length',
        max_length=max_len,
        truncation=False,
        return_tensors='pt'
    )
    pixel_values = proc_inputs.pixel_values.squeeze(0)
    input_ids = proc_inputs.input_ids.squeeze(0)
    attention_mask = proc_inputs.attention_mask.squeeze(0)

    # Tokenize output (reasoning + final answer) for labels
    labels = tokenizer(
        example['output'],
        padding='max_length',
        max_length=max_out,
        truncation=True,
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

    data_files = {'train': args.train_file, 'validation': args.val_file}
    ds = load_dataset('json', data_files=data_files)

    print(f"Loading model {args.base_model} (8-bit={args.in_8bit})...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        load_in_8bit=args.in_8bit,
        device_map='auto',
        torch_dtype=torch.float16 if args.fp16 else None,
    )

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=args.lora_dropout,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg)

    print("Tokenizing and processing images...")
    ds = ds.map(
        lambda ex: preprocess(ex, processor),
        remove_columns=['id', 'source_images', 'instruction', 'output', 'ground_truth_option'],
        batched=False
    )

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

    trainer.train()
    print(f"Saving LoRA model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
