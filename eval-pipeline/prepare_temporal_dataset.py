# -*- coding: utf-8 -*-
"""prepare_temporal_dataset.py  ── v2.1  🛠️

Creates a JSONL ready for **LLaVA‑Interleave‑Qwen‑7B** fine‑tuning.

🔄 **v2.1 changes**
-------------------
* **💥 Fixed key mismatch** – `transformers` chat template expects the field
  `role` instead of `from`.  We now write
  `{"role": "user"|"assistant"|"system", "content": "…"}` so the self‑check
  and the trainer work without the `jinja2.exceptions.UndefinedError` you hit.
* Helper `map_from_to_role()` centralises the mapping.
* Added `--debug_first_n` to run the OpenAI loop on only the first *n* samples
  (handy for cheap quick tests).

Example run
-----------
```bash
python prepare_temporal_dataset.py \ 
  --mmiu_hf_dataset_name FanqingM/MMIU-Benchmark \
  --task_name temporal_ordering \
  --image_base_dir ./data \
  --output_dir ./data/finetuning_data \
  --train_split_size 150 \
  --openai_api_key $OPENAI_KEY \
  --self_check true \
  --debug_first_n 5   # optional quick smoke‑test
```
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ---------- constants ----------

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
SYSTEM_FOR_GPT = (
    "You are an expert in analysing sequences of images to determine their temporal order. "
    "Respond with ONE short rationale sentence and end with the correct letter, "
    "formatted exactly as 'Final Answer: X'."
)
RE_FINAL = re.compile(r"final answer:\s*([A-D])", re.I)

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mmiu_hf_dataset_name", default="FanqingM/MMIU-Benchmark")
    p.add_argument("--mmiu_hf_dataset_config", default=None)
    p.add_argument("--mmiu_hf_dataset_split", default="test")
    p.add_argument("--task_name", default="temporal_ordering")
    p.add_argument("--image_base_dir", required=True,
                   help="Directory that contains Continuous-temporal/ …")
    p.add_argument("--output_dir", default="./finetuning_data")
    p.add_argument("--train_split_size", type=int, default=150)
    p.add_argument("--openai_api_key", default=None)
    p.add_argument("--openai_base_url", default=None)
    p.add_argument("--generator_model", default="gpt-4o")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--self_check", type=lambda s: s.lower() in {"1", "true", "yes"}, default=False)
    p.add_argument("--debug_first_n", type=int, default=None,
                   help="Process only N samples for quick testing")
    return p.parse_args()

# ---------- helpers ----------

def encode_b64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode()

def image_blocks(abs_paths: List[Path]):
    return [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_b64(p)}"}
    } for p in abs_paths]

def map_from_to_role(sentence_list: List[Dict[str, str]]):
    """Convert keys from→role for HF chat template."""
    mapping = {"human": "user", "gpt": "assistant", "system": "system"}
    return [{"role": mapping[s["from"]], "content": s["value"]} for s in sentence_list]

# ---------- OpenAI ----------

def call_openai(client: OpenAI, model: str, human_prompt: str, abs_img_paths: List[Path]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_FOR_GPT},
        {"role": "user", "content": image_blocks(abs_img_paths) + [{"type": "text", "text": human_prompt}]}
    ]
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=256, temperature=0.2)
    return resp.choices[0].message.content.strip()

# ---------- self‑check ----------

def llava_roundtrip(record: Dict, img_root: Path):
    """Run a tokeniser‑roundtrip to ensure prompt + images align.
    Raises if the record is malformed. Safe to skip in production by
    omitting --self_check.
    """
    from transformers import LlavaProcessor
    from PIL import Image

    proc = LlavaProcessor.from_pretrained(
        "llava-hf/llava-interleave-qwen-7b-hf", use_fast=False
    )
    imgs = [Image.open(img_root / p) for p in record["images"]]

    prompt = proc.apply_chat_template(
        record["conversations"], add_generation_prompt=False
    )
    # NOTE: pass the text explicitly via the `text` kwarg, otherwise Python
    # thinks the positional string is meant for the `images` parameter as well,
    # causing: TypeError: __call__() got multiple values for argument 'images'
    proc(text=prompt, images=imgs, return_tensors="pt")

# ---------- prompt builder ----------

def build_human_prompt(sample: dict) -> str:
    placeholders = "\n".join(["<image>" for _ in sample["input_image_path"]])
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    opts = sample.get("options", "")
    return f"{placeholders}\nContext: {ctx}\n{q}\nChoices:\n{opts}"

# ---------- main ----------

if __name__ == "__main__":
    args = parse_args()
    img_root = Path(os.path.abspath(args.image_base_dir))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load dataset
    ds = load_dataset(args.mmiu_hf_dataset_name, name=args.mmiu_hf_dataset_config, split=args.mmiu_hf_dataset_split)
    subset = [r for r in ds if r.get("task", "").lower() == args.task_name.lower()][: args.train_split_size]
    if args.max_samples:
        subset = subset[: args.max_samples]

    # 2) openai client
    key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key required")
    client = OpenAI(api_key=key, base_url=args.openai_base_url)

    out_path = out_dir / "llava_temporal_train.jsonl"
    written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        iterable = subset
        if args.debug_first_n:
            iterable = iterable[: args.debug_first_n]
        for samp in tqdm(iterable, desc="Building dataset"):
            rel_paths = samp["input_image_path"]
            abs_paths = [img_root / p for p in rel_paths]
            if not all(p.exists() for p in abs_paths):
                print(f"[SKIP] missing image for id={samp.get('id')}")
                continue

            human_prompt = build_human_prompt(samp)
            assistant_resp = call_openai(client, args.generator_model, human_prompt, abs_paths)

            match = RE_FINAL.search(assistant_resp)
            letter = match.group(1).upper() if match else None
            gt = str(samp.get("output", "")).strip().upper()
            if letter != gt:
                print(f"[WARN] mismatch assistant {letter} vs GT {gt} — skip")
                continue

            conv = [
                {"role": "system",    "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user",      "content": human_prompt},
                {"role": "assistant", "content": assistant_resp},
            ]

            record = {
                "system_prompt": DEFAULT_SYSTEM_PROMPT,  # legacy
                "images": rel_paths,
                "conversations": conv,
                "ground_truth_option": gt,
            }

            if args.self_check:
                llava_roundtrip(record, img_root)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n✔ Finished. {written} samples → {out_path}")
