import argparse
import os
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


def preprocess(example, processor, max_in=512, max_out=256):
    # expects example contains:
    # - example['image_paths']: list of image file paths
    # - example['instruction']: prompt string with <image> tokens
    # - example['target']: target string (reasoning + answer)
    images = []
    for img_path in example['image_paths']:
        # load and process each image
        pixel = processor(image_path=img_path, return_tensors='pt').pixel_values
        images.append(pixel)
    # concatenate across channel dimension
    pixel_values = torch.cat(images, dim=1).squeeze(0)

    tokenized_inputs = processor.tokenizer(
        example['instruction'],
        truncation=True,
        padding='max_length',
        max_length=max_in,
        return_tensors='pt',
    )
    tokenized_labels = processor.tokenizer(
        example['target'],
        truncation=True,
        padding='max_length',
        max_length=max_out,
        return_tensors='pt',
    )

    input_ids = tokenized_inputs.input_ids.squeeze(0)
    attention_mask = tokenized_inputs.attention_mask.squeeze(0)
    labels = tokenized_labels.input_ids.squeeze(0)

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def main():
    args = parse_args()

    # Load dataset
    data_files = { 'train': args.train_file, 'validation': args.val_file }
    ds = load_dataset('json', data_files=data_files)

    # Load processor & model
    print(f"Loading model {args.base_model} (8-bit={args.in_8bit})...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        load_in_8bit=args.in_8bit,
        device_map='auto'
    )

    # Wrap in LoRA
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=args.lora_dropout,
        task_type=TaskType.VISION_SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_cfg)

    # Preprocess
    print("Tokenizing and processing images...")
    ds = ds.map(
        lambda ex: preprocess(ex, processor),
        remove_columns=ds['train'].column_names
    )

    # Training arguments
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        data_collator=default_data_collator,
    )

    # Train
    trainer.train()
    print("Saving LoRA-adapted model to", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
