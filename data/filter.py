"""Filter raw {prompt, source, target, category} pairs for fidelity + sanity.

Hard rules (drop on violation):
  - Empty source or target
  - Compression ratio (tok(target)/tok(source)) outside the category's allowed band
  - Any code fence content in source not present byte-exact in target (mutated code)
  - For dialogue rows (with `### user` / `### assistant` headings): fence check per
    assistant turn; also verify every user turn from source appears verbatim in target.
  - Target article density >5%
  - Duplicate (source-hash)
  - Chat-templated length >3800 tokens (leave headroom under max_seq_length=4096)

Category compression bands (TRAINING DATA filter — wider than eval gates).
The model needs to *see* compression; we reject expansion and outright-too-short cases.
Semantic-sim gate during eval is the real info-loss guard, not these bounds.

  qa:                       0.10 - 0.85
  review, debug, refactor:  0.20 - 0.85
  dialogue:                 0.30 - 0.90

Run:
    python data/filter.py --in data/out/raw_pairs.jsonl --out data/out/clean_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-.]*\n?(.*?)```", re.DOTALL)
ARTICLE_RE = re.compile(r"\b(the|a|an|is|are|was|were)\b", re.IGNORECASE)
WORD_RE = re.compile(r"\b\w+\b")
TURN_RE = re.compile(r"^### (user|assistant)\s*$", re.MULTILINE)

COMPRESSION_BANDS = {
    "qa":       (0.10, 0.85),
    "review":   (0.20, 0.85),
    "debug":    (0.20, 0.85),
    "refactor": (0.20, 0.85),
    "dialogue": (0.30, 0.90),
}


def tok_count(text: str) -> int:
    try:
        import tiktoken  # type: ignore
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(WORD_RE.findall(text))


def extract_fences(text: str) -> list[str]:
    return [m.group(1).rstrip("\n") for m in FENCE_RE.finditer(text)]


def fences_preserved_whole(source: str, target: str) -> bool:
    src_fences = [f for f in extract_fences(source) if f.strip()]
    return all(f in target for f in src_fences) if src_fences else True


def split_turns(text: str) -> list[tuple[str, str]]:
    """Return [(role, content), ...] for a `### user\\n... ### assistant\\n...` block.
    Returns empty list if the text isn't dialogue-shaped."""
    matches = list(TURN_RE.finditer(text))
    if not matches:
        return []
    turns: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        role = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turns.append((role, text[start:end].strip()))
    return turns


def dialogue_fences_preserved(source: str, target: str) -> bool:
    src_turns = split_turns(source)
    tgt_turns = split_turns(target)
    if not src_turns:
        return fences_preserved_whole(source, target)
    if len(src_turns) != len(tgt_turns):
        return False
    for (sr, sc), (tr, tc) in zip(src_turns, tgt_turns, strict=False):
        if sr != tr:
            return False
        if sr == "user":
            # User turns must pass through byte-exact.
            if sc != tc:
                return False
        else:
            if not fences_preserved_whole(sc, tc):
                return False
    return True


def article_density(text: str) -> float:
    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(1 for _ in ARTICLE_RE.finditer(text)) / len(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rejected", type=Path,
                    default=ROOT / "data" / "out" / "rejected.jsonl")
    ap.add_argument("--max-article-density", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=3800)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    by_cat_kept: Counter[str] = Counter()
    seen: set[str] = set()
    kept: list[dict] = []
    rejected: list[dict] = []

    for line in args.inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            counts["bad_json"] += 1
            continue

        src = (rec.get("source") or "").strip()
        tgt = (rec.get("target") or "").strip()
        prompt = rec.get("prompt") or ""
        category = (rec.get("category") or "qa").lower()
        band = COMPRESSION_BANDS.get(category, COMPRESSION_BANDS["qa"])

        reason: str | None = None
        if not src or not tgt:
            reason = "empty"
        else:
            key = sha1((prompt + "::" + src).encode("utf-8")).hexdigest()
            if key in seen:
                reason = "duplicate"
            else:
                s_tok = tok_count(src)
                t_tok = tok_count(tgt)
                if s_tok == 0:
                    reason = "empty_source"
                else:
                    ratio = t_tok / s_tok
                    if not (band[0] <= ratio <= band[1]):
                        reason = f"compression_out_of_band({ratio:.2f}_vs_{band})"
                    elif s_tok + t_tok > args.max_tokens:
                        reason = "too_long"
                    elif category == "dialogue":
                        if not dialogue_fences_preserved(src, tgt):
                            reason = "dialogue_integrity_violated"
                    else:
                        if not fences_preserved_whole(src, tgt):
                            reason = "code_fence_mutated"
                    if not reason and article_density(tgt) > args.max_article_density:
                        reason = "article_density_high"
                if not reason:
                    seen.add(key)

        counts[reason or "kept"] += 1
        if reason:
            rec["_reject_reason"] = reason
            rejected.append(rec)
        else:
            kept.append(rec)
            by_cat_kept[category] += 1

    with args.out.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with args.rejected.open("w", encoding="utf-8") as f:
        for rec in rejected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"input: {total}")
    for k, v in counts.most_common():
        print(f"  {k:36s} {v:5d}  ({v/total*100:.1f}%)")
    print("kept by category:")
    for c, n in by_cat_kept.most_common():
        print(f"  {c:12s} {n}")
    print(f"kept     -> {args.out.resolve()}")
    print(f"rejected -> {args.rejected.resolve()}")


if __name__ == "__main__":
    main()
