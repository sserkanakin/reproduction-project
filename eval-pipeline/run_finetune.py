import json
from pathlib import Path
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

# 1. load dataset
ds = load_dataset("json", data_files={"train": "train.jsonl", "test": "val.jsonl"})
# 2. load processor & model
processor = AutoProcessor.from_pretrained("llava-hf/llava-interleave-qwen-7b-hf")
model = AutoModelForVision2Seq.from_pretrained(
    "llava-hf/llava-interleave-qwen-7b-hf",
    load_in_8bit=True,  # if using bitsandbytes
    device_map="auto",
)

# 3. wrap in LoRA
lora_cfg = LoraConfig(
    r=8,  # bottleneck rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.VISION_SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_cfg)


# 4. preprocess function
def preprocess(example):
    images = [processor(image_path, return_tensors="pt").pixel_values for image_path in example["image_paths"]]
    # stack along image dimension:
    pixel_values = torch.cat(images, dim=1)
    inputs = processor.tokenizer(
        example["instruction"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    labels = processor.tokenizer(
        example["target"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
    )["input_ids"]
    inputs["labels"] = labels
    inputs["pixel_values"] = pixel_values.squeeze(0)
    return inputs


tokenized = ds.map(preprocess, batched=False)

# 5. trainer
training_args = TrainingArguments(
    output_dir="lora-output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=default_data_collator,
)

trainer.train()
model.save_pretrained("llava-lora-finetuned")
