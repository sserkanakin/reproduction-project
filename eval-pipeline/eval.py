#!/usr/bin/env python3
import json, subprocess, re
from pathlib import Path

TEST_JSON = Path("eval-pipeline/data/finetuning_data/test.jsonl")
MODEL_PATH = "checkpoints/llava-merged"
MODEL_BASE = "llava-hf/llava-interleave-qwen-0.5b-hf"

total = correct = 0

with TEST_JSON.open() as f:
    for line in f:
        sample = json.loads(line)
        imgs = ",".join(sample["images"])
        context = sample.get("context","")
        question = sample.get("question","")
        options = sample.get("options","")
        # build prompt
        prompt = "\n".join([
            f"Context: {context}",
            f"Question: {question}",
            f"Choices:\n{options}",
            "Answer the letter (A/B/C/D) directly."
        ])
        # call run_llava
        cmd = [
            "python","run_llava.py",
            "--model-path", MODEL_PATH,
            "--model-base", MODEL_BASE,
            "--image-file", imgs,
            "--query", prompt,
            "--temperature","0",
            "--num_beams","1",
            "--max_new_tokens","4"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        raw = proc.stdout.strip()
        # extract choice
        m = re.match(r"\s*([A-DZ])", raw, re.IGNORECASE)
        choice = (m.group(1).upper() if m else raw[:1].upper())
        gt = sample.get("output","").strip().upper()

        total += 1
        if choice == gt:
            correct += 1
        print(f"[{total}] ID={sample.get('id')} GT={gt} PRED={choice} RAW=\"{raw}\"")

print()
print(f"Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
