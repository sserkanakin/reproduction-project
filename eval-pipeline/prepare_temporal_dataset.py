#!/usr/bin/env python3
"""
prepare_temporal_dataset.py – build LLaVA-ready temporal-ordering data
----------------------------------------------------------------------
✓ Adds the mandatory universal <image> token
✓ Saves **train.json / test.json** as single JSON arrays (not JSONL)
"""

from __future__ import annotations
import argparse, json, os, random, re, sys
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_IMAGE_TOKEN = "<image>"          # LLaVA expects this
SEP = "\n"

# ----------------------------------------------------------------------------- helpers
def parse_options(block: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for line in block.strip().splitlines():
        m = re.match(r"([A-Z]):\s*\[(.*)\]", line.strip())
        if m:
            out[m[1]] = [int(i) for i in m[2].split(",")]
    return out


def build_user_prompt(images: List[str], question: str) -> str:
    """First line must be the DEFAULT_IMAGE_TOKEN."""
    placeholders = [f"<image_{i}>" for i in range(len(images))]
    return SEP.join([DEFAULT_IMAGE_TOKEN, *placeholders, "", question])


def convert_sample(raw: Dict[str, Any], add_reasoning: bool = False) -> Dict[str, Any]:
    option_map = parse_options(raw["options"])
    gt_order   = option_map[raw["output"].strip()]

    assistant_text = f"The correct order is {gt_order}."
    if add_reasoning:
        assistant_text += "\n\n**Reasoning**: (placeholder)"

    return {
        "id": Path(raw["input_image_path"][0]).stem,
        "images": raw["input_image_path"],
        "conversations": [
            {"from": "human", "value": build_user_prompt(raw["input_image_path"], raw["question"])},
            {"from": "gpt",   "value": assistant_text},
        ],
    }

# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mmiu_file",  type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--train_size", type=int,  default=150)
    p.add_argument("--seed",       type=int,  default=42)
    args = p.parse_args()

    raw = [json.loads(l) for l in open(args.mmiu_file, encoding="utf-8")]
    random.seed(args.seed); random.shuffle(raw)
    train_raw, test_raw = raw[:args.train_size], raw[args.train_size:]

    train = [convert_sample(s, add_reasoning=True)  for s in train_raw]
    test  = [convert_sample(s)                      for s in test_raw]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2))
    (args.output_dir / "test.json").write_text(json.dumps(test,  ensure_ascii=False, indent=2))
    print(f"✓ Wrote {len(train)} train / {len(test)} test samples to {args.output_dir}")

if __name__ == "__main__":
    main()
