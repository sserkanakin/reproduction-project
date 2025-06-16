#!/usr/bin/env python3
"""
prepare_temporal_dataset.py  –  adds real reasoning with GPT-4o

• Reads MMIU file (.json or .jsonl)
• Adds <image> + <image_0> … placeholders
• Generates chain-of-thought via OpenAI if OPENAI_API_KEY is set
• Writes train.json (with reasoning) / test.json (no reasoning)
"""
from __future__ import annotations
import argparse, json, os, random, re
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
import tiktoken

load_dotenv()  # Load environment variables from a .env file

DEFAULT_IMAGE_TOKEN = "<image>"
SEP = "\n"

# --------------------------------------------------------------------------- helpers
def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def truncate_text(text: str, model: str, max_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)

def parse_options(block: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for ln in block.strip().splitlines():
        if (m := re.match(r"([A-Z]):\s*\[(.*)\]", ln.strip())):
            out[m[1]] = [int(x) for x in m[2].split(",")]
    return out

def build_prompt(imgs: list[str], q: str) -> str:
    return SEP.join([DEFAULT_IMAGE_TOKEN, *[f"<image_{i}>" for i in range(len(imgs))], "", q])

def call_openai_reasoning(gt: List[int], sample: dict[str, Any], model: str) -> str:
    """Ask GPT-4o to explain the ordering."""
    import openai, textwrap
    client = openai.OpenAI()   # uses env var OPENAI_API_KEY
    ordering = ", ".join(map(str, gt))
    user_msg = textwrap.dedent(f"""\
        The correct order of the {len(gt)} frames is {ordering}.
        Explain step-by-step how the visual evidence in each frame supports
        this temporal sequence. Mention concrete cues (e.g. body posture,
        motion trails, object state) rather than generic statements. You do not need to create a any tables. Give you full reasoning and final answer.
    """)
    # Truncate prompt to ensure it does not exceed 300 tokens.
    max_prompt_tokens = 300
    if count_tokens(user_msg, model) > max_prompt_tokens:
        user_msg = truncate_text(user_msg, model, max_prompt_tokens)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.3,
        max_tokens=2048,  # Adjust as needed
    )
    return resp.choices[0].message.content.strip()

def convert(sample: dict[str, Any], add_reason: bool, model: str) -> dict[str, Any]:
    order = parse_options(sample["options"])[sample["output"].strip()]
    answer = f"The correct order is {order}."
    if add_reason:
        try:
            answer += "\n\n" + call_openai_reasoning(order, sample, model)
        except Exception as e:
            answer += f"\n\n**Reasoning**: (fallback – {e})"
    return {
        "id": Path(sample["input_image_path"][0]).stem,
        "images": sample["input_image_path"],
        "conversations": [
            {"from": "human", "value": build_prompt(sample["input_image_path"], sample["question"])},
            {"from": "gpt",   "value": answer},
        ],
    }

def load(path: Path) -> list[dict[str, Any]]:
    txt = path.read_text(encoding="utf-8").lstrip()
    return json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]

# --------------------------------------------------------------------------- main
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--mmiu_file",   type=Path, required=True)
    pa.add_argument("--output_dir",  type=Path, required=True)
    pa.add_argument("--train_size",  type=int,  default=150)
    pa.add_argument("--seed",        type=int,  default=42)
    pa.add_argument("--reasoning_model", type=str, default="gpt-4.1",
                    help="OpenAI model for chain-of-thought generation")
    args = pa.parse_args()

    raw = load(args.mmiu_file)
    if len(raw) < args.train_size:
        raise SystemExit(f"train_size {args.train_size} > dataset size {len(raw)}")

    random.seed(args.seed); random.shuffle(raw)
    train_raw, test_raw = raw[:args.train_size], raw[args.train_size:]

    have_key = bool(os.getenv("OPENAI_API_KEY"))
    if not have_key:
        print("⚠️  OPENAI_API_KEY not set – reasoning will be placeholder", flush=True)

    train = [convert(s, add_reason=have_key, model=args.reasoning_model) for s in train_raw]
    test  = [convert(s, add_reason=False,    model=args.reasoning_model) for s in test_raw]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2))
    (args.output_dir / "test.json" ).write_text(json.dumps(test,  ensure_ascii=False, indent=2))
    print(f"✓ Wrote {len(train)} train / {len(test)} test samples to {args.output_dir}")

if __name__ == "__main__":
    main()