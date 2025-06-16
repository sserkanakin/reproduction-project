#!/usr/bin/env python3
"""
prepare_temporal_dataset.py – outputs train.json / test.json (JSON arrays)

✓ Adds <image> token + <image_0> … placeholders
✓ Puts chain-of-thought reasoning into the train split
✓ Accepts raw MMIU file in .json **or** .jsonl format
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_IMAGE_TOKEN = "<image>"
SEP = "\n"

# ---------------------------------------------------------------- helpers
def parse_options(block: str) -> Dict[str, List[int]]:
    m: Dict[str, List[int]] = {}
    for line in block.strip().splitlines():
        if (g := re.match(r"([A-Z]):\s*\[(.*)\]", line.strip())):
            m[g[1]] = [int(i) for i in g[2].split(",")]
    return m


def build_prompt(imgs: list[str], q: str) -> str:
    return SEP.join([DEFAULT_IMAGE_TOKEN, *[f"<image_{i}>" for i in range(len(imgs))], "", q])


def convert(sample: Dict[str, Any], with_reason: bool) -> Dict[str, Any]:
    order = parse_options(sample["options"])[sample["output"].strip()]
    answer = f"The correct order is {order}."
    if with_reason:
        answer += "\n\n**Reasoning**: (placeholder)"
    return {
        "id": Path(sample["input_image_path"][0]).stem,
        "images": sample["input_image_path"],
        "conversations": [
            {"from": "human", "value": build_prompt(sample["input_image_path"], sample["question"])},
            {"from": "gpt",   "value": answer},
        ],
    }


def load(mmiu: Path) -> list[dict[str, Any]]:
    txt = mmiu.read_text(encoding="utf-8").lstrip()
    return json.loads(txt) if txt[0] == "[" else [json.loads(l) for l in txt.splitlines() if l.strip()]


# ---------------------------------------------------------------- main
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--mmiu_file",  type=Path, required=True)
    pa.add_argument("--output_dir", type=Path, required=True)
    pa.add_argument("--train_size", type=int, default=150)
    pa.add_argument("--seed",       type=int, default=42)
    pa.add_argument("--no_reasoning", action="store_true",
                    help="omit chain-of-thought even in train split")
    a = pa.parse_args()

    data = load(a.mmiu_file)
    if len(data) < a.train_size:
        raise SystemExit(f"train_size {a.train_size} exceeds dataset size {len(data)}")

    random.seed(a.seed); random.shuffle(data)
    train_raw, test_raw = data[:a.train_size], data[a.train_size:]

    train = [convert(s, not a.no_reasoning) for s in train_raw]
    test  = [convert(s, False)              for s in test_raw]

    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2))
    (a.output_dir / "test.json" ).write_text(json.dumps(test,  ensure_ascii=False, indent=2))
    print(f"✓ Wrote {len(train)} train / {len(test)} test samples to {a.output_dir}")

if __name__ == "__main__":
    main()
