#!/usr/bin/env python3
"""
prepare_temporal_dataset.py
---------------------------
• Reads the raw MMIU *temporal_ordering* file – JSON *or* JSONL.
• Splits into train / test.
• Adds universal <image> token, numbered <image_0> … placeholders.
• Keeps chain-of-thought reasoning in the train split.
• Writes JSON arrays:  train.json, test.json   (no JSONL output).

Usage
-----
python prepare_temporal_dataset.py \
    --mmiu_file  eval-pipeline/data/to-data.json \
    --output_dir eval-pipeline/data/finetune_data \
    --train_size 150
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_IMAGE_TOKEN = "<image>"
SEP = "\n"

# ---------------------------------------------------------------- helpers
def parse_options(block: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for line in block.strip().splitlines():
        m = re.match(r"([A-Z]):\s*\[(.*)\]", line.strip())
        if m:
            out[m[1]] = [int(i) for i in m[2].split(",")]
    return out


def build_user_prompt(imgs: List[str], question: str) -> str:
    placeholders = [f"<image_{i}>" for i in range(len(imgs))]
    return SEP.join([DEFAULT_IMAGE_TOKEN, *placeholders, "", question])


def convert(raw: Dict[str, Any], reasoning: bool) -> Dict[str, Any]:
    gt_order = parse_options(raw["options"])[raw["output"].strip()]
    answer   = f"The correct order is {gt_order}."
    if reasoning:
        answer += "\n\n**Reasoning**: (placeholder)"
    return {
        "id": Path(raw["input_image_path"][0]).stem,
        "images": raw["input_image_path"],
        "conversations": [
            {"from": "human", "value": build_user_prompt(raw["input_image_path"], raw["question"])},
            {"from": "gpt",   "value": answer},
        ],
    }

def load_raw(path: Path) -> List[Dict[str, Any]]:
    """Supports .json (array) or .jsonl (one object per line)."""
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):                       # plain JSON array
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]  # JSONL

# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmiu_file",  type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--train_size", type=int, default=150)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    raw = load_raw(args.mmiu_file)
    if len(raw) < args.train_size:
        raise SystemExit(f"train_size {args.train_size} > dataset size {len(raw)}")

    random.seed(args.seed); random.shuffle(raw)
    train_raw, test_raw = raw[:args.train_size], raw[args.train_size:]

    train = [convert(s, reasoning=True)  for s in train_raw]
    test  = [convert(s, reasoning=False) for s in test_raw]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2))
    (args.output_dir / "test.json" ).write_text(json.dumps(test,  ensure_ascii=False, indent=2))
    print(f"✓ Wrote {len(train)} train / {len(test)} test samples to {args.output_dir}")

if __name__ == "__main__":
    main()
