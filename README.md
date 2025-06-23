# MMIU Multi-Image Understanding Pipeline

This repository provides a robust pipeline for preparing data, fine-tuning, and benchmarking multi-image understanding models such as **Idefics2** and **LLaVA**. The workflow is designed for research and experimentation with temporal and reasoning tasks involving sequences of images and natural language.

---

## 🚀 Overview

The pipeline consists of three main stages:

1. **Dataset Preparation**  
   Generate training, evaluation, and test datasets with step-by-step explanations using the OpenAI Vision API.

2. **Fine-Tuning**  
   Fine-tune the Idefics2 model using LoRA adapters and a custom data collator for interleaved image-text conversations.

3. **Benchmarking**  
   Evaluate both base and fine-tuned models on complex multi-image tasks, using OpenAI models for robust answer parsing and accuracy reporting.

---

## 🛠️ Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sserkanakin/reproduction-project.git
   cd reproduction-project
   ```

2. **Install Conda (if not already installed)**  
   [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html)

3. **Create and Activate the Environment**
   ```bash
   conda env create -f environment.yml
   conda activate tune
   ```

   This will install all dependencies listed in `requirements.txt` via pip.

4. **Configure OpenAI API Key**  
   Create a `.env` file in `src/` with your OpenAI API key:
   ```
   OPENAI_API_KEY="sk-your-key"
   ```

---

## 📁 Folder Structure

```
src/
  scripts/
    prepare_temporal_dataset.py   # Prepare datasets with OpenAI explanations
    finetune_idefics2.py          # Fine-tune Idefics2 with LoRA
    benchmark_idefics2.py         # Benchmark Idefics2 (base & fine-tuned)
    benchmark-llava.py            # Benchmark LLaVA models
```

---

## 1️⃣ Dataset Preparation

Prepare temporal ordering datasets with explanations:

```bash
python src/scripts/prepare_temporal_dataset.py \
  --mmiu_file data/to-data.json \
  --output_dir data/finetune_data/ \
  --train_size 100
```

- **Inputs:**  
  - `to-data.json`: Source data with image paths and options.
- **Outputs:**  
  - `train.json`: Training data with OpenAI-generated explanations.
  - `eval.json`: Evaluation data in the same format.
  - `test.json`: Unchanged test samples.

---

## 2️⃣ Fine-Tuning

### ⚡️ Idefics2

Fine-tune the Idefics2-8b model using LoRA and the prepared dataset:

```bash
python src/scripts/finetune_idefics2.py \
  --train_file data/finetune_data/train.json \
  --val_file data/finetune_data/eval.json \
  --image_base_path data/finetune_data/ \
  --output_dir idefics2-8b-temporal-finetune \
  --epochs 1
```

- **Features:**
  - Custom data collator for robust image-text interleaving.
  - LoRA parameter-efficient fine-tuning.
  - Gradient checkpointing and 8-bit optimizer support.

---

### ⚡️ LLaVA

To fine-tune LLaVA, follow these steps:

1. **Clone the lmms-finetune repository:**

   ```bash
   git clone https://github.com/zjysteven/lmms-finetune.git
   ```

2. **Download or prepare the required LLaVA model checkpoints** 

   After the steps taken to create the final checkpoints for the tuned LLaVA model, folow these steps:

   Place the model folders as follows (relative to your project root):

   ```
   src/eval_pipeline/lmms-finetune/checkpoints/llava-interleave-qwen-7b-merged/
   src/eval_pipeline/lmms-finetune/checkpoints/llava-interleave-qwen-0.5b-merged/  # (if using 0.5b)
   ```

   - The script expects the **finetuned model** at:
     ```
     src/eval_pipeline/lmms-finetune/checkpoints/llava-interleave-qwen-7b-merged/
     ```
   - The **base model** is referenced as:
     ```
     llava-hf/llava-interleave-qwen-7b-hf
     ```
     (This can be downloaded automatically by Hugging Face Transformers if not present locally.)

3. **Ensure your test data is at:**

   ```
   src/eval_pipeline/data/finetune_data/test.json
   ```

---

## 3️⃣ Benchmarking

### Idefics2

Evaluate both the base and fine-tuned Idefics2 models:

```bash
python src/scripts/benchmark_idefics2.py \
  --val_file data/finetune_data/test.json \
  --image_base_path data/ \
  --adapter_path idefics2-8b-temporal-finetune/final_checkpoint \
  --results_json_path results/results.json \
  --summary_txt_path results/summary.txt
```

- **Outputs:**
  - `results.json`: Detailed per-sample results.
  - `summary.txt`: Accuracy summary.

### LLaVA

Benchmark LLaVA models:

```bash
python src/scripts/benchmark-llava.py
```
(model and data paths can be modified in the script)

---

## 🧠 What Does This Pipeline Do?

- **Automates** the creation of multi-image, multi-turn datasets with rich explanations.
- **Fine-tunes** state-of-the-art vision-language models for temporal and reasoning tasks.
- **Benchmarks** model performance using robust, OpenAI-powered answer parsing.
- **Supports** research in multi-image understanding, temporal reasoning, and explainable AI.

---

## 📌 Notes

- Ensure all image paths in your data are correct and accessible.
- The OpenAI API is used for both explanation generation and answer parsing—API usage costs may apply.
- For large-scale runs, consider batching and monitoring API usage.

---

## 📄 License

MIT License (see `LICENSE` file).