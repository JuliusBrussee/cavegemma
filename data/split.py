"""Deterministic 90/10 train/eval split on prompt-hash, with seed-pair pinning.

Seed pairs from extract_seeds.py (origin starts with 'seed:') always go to train —
they're hand-authored gold and too few to spare for eval.

Run:
    python data/split.py --in data/out/clean_pairs.jsonl --seeds data/out/seed_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def hash_prompt(prompt: str) -> int:
    return int(sha1(prompt.encode("utf-8")).hexdigest(), 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--seeds", type=Path, default=ROOT / "data" / "out" / "seed_pairs.jsonl")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "out" / "train.jsonl")
    ap.add_argument("--eval", dest="ev", type=Path, default=ROOT / "data" / "out" / "eval.jsonl")
    ap.add_argument("--eval-frac", type=float, default=0.10)
    args = ap.parse_args()

    args.train.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for line in args.inp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    seed_records: list[dict] = []
    if args.seeds.exists():
        for line in args.seeds.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seed_records.append(json.loads(line))

    train: list[dict] = list(seed_records)  # all seeds -> train
    ev: list[dict] = []

    cutoff = int(args.eval_frac * (1 << 64))
    for rec in records:
        h = hash_prompt(rec.get("prompt", "") + "::" + rec.get("source", ""))
        bucket = h % (1 << 64)
        (ev if bucket < cutoff else train).append(rec)

    with args.train.open("w", encoding="utf-8") as f:
        for rec in train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with args.ev.open("w", encoding="utf-8") as f:
        for rec in ev:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"train: {len(train)}  (incl. {len(seed_records)} seed pairs)")
    print(f"eval:  {len(ev)}")


if __name__ == "__main__":
    main()
