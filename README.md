# MMIU LLaVA Evaluation Pipeline

This repository contains a small collection of scripts used to fine tune and evaluate multi‑image understanding models such as LLaVA and Idefics2.  The code was originally prepared for Docker but has been simplified to run directly in a Conda environment.

## Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
2. Create the environment and install packages:

   ```bash
   conda env create -f environment.yml
   conda activate llava-env
   ```

   The `environment.yml` file installs all Python packages listed in `requirements.txt` using `pip`.

3. Copy your OpenAI API key into `src/.env`:

   ```env
   OPENAI_API_KEY="sk-your-key"
   ```

   This file is ignored by git.

## Running the Pipeline

### Benchmarking

Run the benchmark script on the included sample dataset:
TODO

```bash
python src/benchmark_idefics2.py \
  --val_file data/finetune_data/test.json \
  --image_base_path data/ \
  --adapter_path path/to/adapter \
  --results_json_path results/results.json \
  --summary_txt_path results/summary.txt
```

### Simple Inference

A minimal multi‑image inference example is provided:

```bash
python src/run_inference.py
```

Adjust model IDs and image paths inside the script as needed.

## Notes

The data under `src/eval_pipeline/data` is a small sample to demonstrate the file format. For real experiments you will need to supply the full dataset and image files.
