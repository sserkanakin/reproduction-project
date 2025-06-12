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
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning script for LLaVA models.")
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


class FineTuningDataset(Dataset):
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
                    return None
        try:
            inputs = self.processor(
                text=full_text, images=images, return_tensors="pt",
                padding="max_length", truncation=True, max_length=self.max_length
            )
            inputs['labels'] = inputs['input_ids'].clone()
            return {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception:
            return None


def custom_data_collator(features):
    return {key: torch.stack([f[key] for f in features]) for key in features[0].keys()} if features else {}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    quant_config = BitsAndBytesConfig(load_in_8bit=True) if args.use_quantization else None
    model = LlavaForConditionalGeneration.from_pretrained(args.base_model_id, torch_dtype=torch.float16,
                                                          quantization_config=quant_config, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    lora_config = LoraConfig(r=16, lora_alpha=32,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                             "down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = FineTuningDataset(args.finetuning_dataset_path, processor, args.image_base_dir, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, logging_steps=args.logging_steps,
        save_steps=args.save_steps, max_steps=args.max_steps, fp16=True,
        optim="paged_adamw_8bit", save_total_limit=2, remove_unused_columns=False, report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=custom_data_collator)
    trainer.train()

    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_checkpoint_dir)
    print(f"\n--- Fine-tuning Complete --- \nFinal LoRA adapters saved to: {final_checkpoint_dir}")


if __name__ == "__main__":
    main()