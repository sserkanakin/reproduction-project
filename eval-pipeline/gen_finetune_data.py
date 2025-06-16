#!/usr/bin/env python3
"""
prepare_temporal_dataset.py
--------------------------------
Utility for converting MMIU *temporal_ordering* items into the JSONL
conversation format expected by LLaVA‑Interleave models.

The script will:
1. Load the raw MMIU JSON/JSONL file.
2. Randomly split it into a train and test partition (default 150 / 50).
3. For **train** samples it can optionally call an external LLM (e.g. GPT‑4o)
   to produce step‑by‑step reasoning that explains the correct temporal order.
   This produces richer supervision for chain‑of‑thought fine‑tuning.
4. Emit two files – ``train.jsonl`` and ``test.jsonl`` – in which each line is
   a JSON object of the form::

      {
        "id": "temporal_ordering_84",
        "images": [
          "./Continuous-temporal/temporal_ordering/temporal_ordering_84_0.jpg",
          ...
        ],
        "conversations": [
          {
            "from": "human",
            "value": "<image_0>\n<image_1>\n<image_2>\n<image_3>\n<image_4>\n<image_5>\n<image_6>\n\nPlease predict the order of the following pictures by giving each picture an index starting from 0, where larger indices indicate later moments."
          },
          {
            "from": "gpt",
            "value": "The correct order is [6, 5, 3, 0, 4, 2, 1].\n\n**Reasoning**\n1. … (chain‑of‑thought) …"
          }
        ]
      }

The resulting files are immediately consumable by the stock ``train_mem.py`` /
``finetune_task_lora.sh`` scripts in the `LLaVA‑NeXT` repository when you set
``--data_path train.jsonl``.

Usage
-----
::

    python prepare_temporal_dataset.py \
        --mmiu_file mmiu_temporal.jsonl \
        --output_dir ./llava_temporal_data \
        --train_size 150 \
        --seed 1234 \
        --generate_reasoning \
        --reasoning_model gpt-4o \
        --openai_api_key $OPENAI_API_KEY

Notes
-----
* If ``--generate_reasoning`` is omitted, the script will still create the
  correct answer but without the explanatory chain‑of‑thought.
* You can point ``--reasoning_endpoint`` to any REST endpoint that accepts the
  OpenAI ChatCompletions schema. Only a single request per sample is sent.
* The test split is *always* left without reasoning so it remains a clean
  evaluation set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tqdm

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def parse_options(options_block: str) -> Dict[str, List[int]]:
    """Parse the multiline *options* field from MMIU into a mapping of
    choice‑letter → list‑of‑indices.
    """
    mapping: Dict[str, List[int]] = {}
    for line in options_block.strip().splitlines():
        match = re.match(r"([A-Z]):\s*\[(.*)\]", line.strip())
        if match:
            letter, indices = match.groups()
            mapping[letter] = [int(x) for x in indices.split(',')]
    return mapping


def build_user_prompt(images: List[str], question: str) -> str:
    placeholders = [f"<image_{i}>" for i in range(len(images))]
    return "\n".join(placeholders) + "\n\n" + question


def call_llm_for_reasoning(prompt: str, answer: List[int], model: str) -> str:
    """Stub for calling an external LLM to obtain chain‑of‑thought reasoning.
    Replace this with your preferred provider SDK. The current implementation
    returns a placeholder so that the pipeline remains fully offline‑capable.
    """
    # TODO: integrate with OpenAI SDK or HF Inference endpoints.
    answer_str = ", ".join(map(str, answer))
    return (
        f"The correct order is [{answer_str}].\n\n"
        "**Reasoning (placeholder)**: Detailed step‑by‑step analysis goes here."
    )


def convert_sample(raw: Dict[str, Any], reasoning: bool = False, model: str | None = None) -> Dict[str, Any]:
    images: List[str] = raw["input_image_path"]
    question: str = raw["question"]
    user_prompt: str = build_user_prompt(images, question)

    # Determine the ground‑truth ordering.
    option_map = parse_options(raw["options"])
    gt_letter = raw["output"].strip()
    gt_order = option_map[gt_letter]

    if reasoning and model:
        assistant_response = call_llm_for_reasoning(question, gt_order, model)
    else:
        assistant_response = f"The correct order is {gt_order}."

    return {
        "id": raw.get("id", os.path.splitext(os.path.basename(images[0]))[0]),
        "images": images,
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant_response},
        ],
    }


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare MMIU temporal data for LLaVA fine‑tuning")
    p.add_argument("--mmiu_file", type=Path, required=True, help="Path to the raw MMIU JSON or JSONL file")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to write train/test JSONL files")
    p.add_argument("--train_size", type=int, default=150, help="Number of samples for training (rest used for test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generate_reasoning", action="store_true", help="Whether to create chain‑of‑thought answers")
    p.add_argument("--reasoning_model", type=str, default="gpt-4o", help="Name of the external LLM to query")
    args = p.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data (supports .json or .jsonl)
    with open(args.mmiu_file, "r", encoding="utf‑8") as f:
        if args.mmiu_file.suffix == ".json":
            raw_samples = json.load(f)
        else:
            raw_samples = [json.loads(line) for line in f]

    if len(raw_samples) < args.train_size:
        sys.exit(f"Requested train_size {args.train_size} exceeds dataset size {len(raw_samples)}")

    random.shuffle(raw_samples)
    train_raw = raw_samples[: args.train_size]
    test_raw = raw_samples[args.train_size :]

    # Convert samples
    train_converted = [
        convert_sample(sample, reasoning=args.generate_reasoning, model=args.reasoning_model) for sample in tqdm.tqdm(train_raw, desc="train")
    ]
    test_converted = [convert_sample(sample, reasoning=False) for sample in tqdm.tqdm(test_raw, desc="test")]

    # Save
    train_path = args.output_dir / "train.jsonl"
    test_path = args.output_dir / "test.jsonl"
    with train_path.open("w", encoding="utf‑8") as f:
        for item in train_converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with test_path.open("w", encoding="utf‑8") as f:
        for item in test_converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_converted)} training samples → {train_path}")
    print(f"Wrote {len(test_converted)} test samples     → {test_path}")


if __name__ == "__main__":
    main()
