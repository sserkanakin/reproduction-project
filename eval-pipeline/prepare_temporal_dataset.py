#!/usr/bin/env python3
"""
prepare_temporal_dataset.py  (rev‑C)
===================================
Build a LLaVA‑Interleave fine‑tuning dataset from MMIU *temporal_ordering*.

**What’s new in rev‑C**
-----------------------
The assistant reply now states **both** the chosen option letter (A–D) **and**
the explicit order list before the reasoning, e.g.:

```
The correct option is B. The correct order is [6, 5, 3, 0, 4, 2, 1].

1. …reasoning…
```
This gives the model a direct mapping from option → index list and keeps the
multiple‑choice framing consistent during fine‑tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import tqdm
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_options(block: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for line in block.strip().splitlines():
        m = re.match(r"([A-Z]):\s*\[(.*)\]", line.strip())
        if m:
            letter, indices = m.groups()
            out[letter] = [int(x.strip()) for x in indices.split(',') if x.strip()]
    if not out:
        raise ValueError("Could not parse options block")
    return out


def build_user_prompt(images: List[str], question: str, choices: str) -> str:
    placeholders = "\n".join(f"<image_{i}>" for i in range(len(images)))
    return f"{placeholders}\n\n{question.strip()}\n\n{choices.strip()}"

# -----------------------------------------------------------------------------
# Chain‑of‑thought via OpenAI ≥1.0
# -----------------------------------------------------------------------------

def openai_reasoning(answer: List[int], opt_letter: str, model: str) -> str:
    """Return numbered reasoning or placeholder if API fails/missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    answer_str = ", ".join(map(str, answer))

    if not api_key:
        return (
            f"The correct option is {opt_letter}. The correct order is [{answer_str}].\n\n"
            "**Reasoning (placeholder)**: OPENAI_API_KEY not set."
        )
    try:
        import openai  # >=1.0.0
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.4,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a vision‑language tutor. Explain in 3–6 numbered steps why "
                        "the ground‑truth temporal order is correct, referencing image indices 0‑6."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Ground‑truth option: {opt_letter} → {answer}. Provide the reasoning."
                    ),
                },
            ],
        )
        reasoning = resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        reasoning = f"**Reasoning (placeholder)**: OpenAI call failed – {exc}."

    return (
        f"The correct option is {opt_letter}. The correct order is [{answer_str}].\n\n"
        f"{reasoning}"
    )

# -----------------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------------

def convert_sample(raw: Dict[str, Any], *, add_cot: bool, model: str) -> Dict[str, Any]:
    imgs: List[str] = raw["input_image_path"]
    choices_txt = raw.get("context") or f"Select from the following choices.\n{raw['options']}"
    prompt = build_user_prompt(imgs, raw["question"], choices_txt)

    gt_letter = raw["output"].strip()
    option_map = parse_options(raw["options"])
    gt_order = option_map[gt_letter]

    if add_cot:
        assistant = openai_reasoning(gt_order, gt_letter, model)
    else:
        assistant = (
            f"The correct option is {gt_letter}. The correct order is {gt_order}."
        )

    return {
        "id": raw.get("id", os.path.splitext(os.path.basename(imgs[0]))[0]),
        "images": imgs,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": assistant},
        ],
    }

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Prepare temporal‑ordering for LLaVA‑Interleave")
    ap.add_argument("--mmiu_file", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--train_size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generate_reasoning", action="store_true")
    ap.add_argument("--reasoning_model", default="gpt-4o-mini")
    args = ap.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw MMIU
    with args.mmiu_file.open(encoding="utf-8") as fh:
        raw = json.load(fh) if args.mmiu_file.suffix == ".json" else [json.loads(x) for x in fh]

    if args.train_size > len(raw):
        sys.exit("train_size exceeds dataset size")

    random.shuffle(raw)
    train_raw, test_raw = raw[: args.train_size], raw[args.train_size :]

    for split, subset, cot in (
        ("train", train_raw, args.generate_reasoning),
        ("test", test_raw, False),
    ):
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for sample in tqdm.tqdm(subset, desc=split):
                obj = convert_sample(sample, add_cot=cot, model=args.reasoning_model)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset)} {split} samples → {path}")

if __name__ == "__main__":
    main()
