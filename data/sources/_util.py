"""Shared helpers for source loaders."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from typing import Any

WORD_RE = re.compile(r"\b\w+\b")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def looks_english(text: str) -> bool:
    """Cheap English filter. ~95% precision on technical text."""
    if not text:
        return False
    # rough ASCII letter ratio
    letters = sum(1 for c in text if c.isascii() and c.isalpha())
    return letters >= 0.6 * max(1, len(text))


def has_code(text: str) -> bool:
    return "```" in (text or "")


def take(stream: Iterable[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Cap a record stream to n. Use after filters so partial yields aren't wasted."""
    for i, rec in enumerate(stream):
        if i >= n:
            return
        yield rec


def cli_main(loader_name: str, iter_records_fn) -> None:
    """Shared `python -m data.sources.<name> --n 5` smoke test entrypoint."""
    ap = argparse.ArgumentParser(prog=f"data.sources.{loader_name}")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    for rec in take(iter_records_fn(limit=args.n * 4), args.n):
        # Truncate long fields for display only.
        display = {
            **rec,
            **{
                k: (rec[k][:240] + "…") if isinstance(rec.get(k), str) and len(rec[k]) > 240 else rec.get(k)
                for k in ("prompt", "source_normal", "source_seed")
            },
        }
        print(json.dumps(display, ensure_ascii=False, indent=2))
    print(f"OK ({loader_name})", file=sys.stderr)
