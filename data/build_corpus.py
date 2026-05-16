"""Orchestrate all per-source loaders, dedup, and emit a unified corpus_raw.jsonl.

Quotas (default = v2 plan):
    oasst2          400   (dialogue)
    swe_bench       400   (debug)
    codereview      400   (review)
    commitpack      300   (refactor)
    evol_codealpaca 1200  (qa)
    ultrachat       300   (qa)
                   ----
                   3000

Run:
    python data/build_corpus.py --out data/out/corpus_raw.jsonl
    python data/build_corpus.py --only oasst2 --limit 20    # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# allow `python data/build_corpus.py` (not just `python -m data.build_corpus`)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_QUOTAS = {
    "oasst2": 400,
    "swe_bench": 400,
    "codereview": 400,
    "commitpack": 300,
    "evol_codealpaca": 1200,
    "ultrachat": 300,
}


def _hash_prompt(p: str) -> str:
    return sha1(p.strip().lower().encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "out" / "corpus_raw.jsonl")
    ap.add_argument("--only", help="comma-sep loader names; default = all")
    ap.add_argument("--limit", type=int, default=0, help="override per-source quota")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    sources = (
        args.only.split(",")
        if args.only
        else list(DEFAULT_QUOTAS.keys())
    )

    quotas = {s: (args.limit or DEFAULT_QUOTAS.get(s, 100)) for s in sources}

    seen: set[str] = set()
    written = 0
    per_source: dict[str, int] = dict.fromkeys(sources, 0)

    with args.out.open("w", encoding="utf-8") as f:
        for src in sources:
            try:
                mod = __import__(f"data.sources.{src}", fromlist=["iter_records"])
            except ModuleNotFoundError as e:
                print(f"  ! missing loader: {src} ({e})", file=sys.stderr)
                continue
            print(f">>> loading {src} (quota={quotas[src]})")
            target = quotas[src]
            got = 0
            try:
                for rec in mod.iter_records(limit=target * 2):
                    if got >= target:
                        break
                    h = _hash_prompt(rec["prompt"])
                    if h in seen:
                        continue
                    seen.add(h)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    got += 1
                    written += 1
            except Exception as e:
                print(f"  ! {src} aborted after {got} records: {e}", file=sys.stderr)
            per_source[src] = got
            print(f"    {src}: {got}/{target}")

    print(f"\ntotal written: {written} -> {args.out.resolve()}")
    print("per-source counts:")
    for s, n in per_source.items():
        print(f"  {s:20s} {n}")


if __name__ == "__main__":
    main()
