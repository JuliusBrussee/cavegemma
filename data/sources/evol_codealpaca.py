"""theblackcat102/evol-codealpaca-v1 (Apache-2.0 fork) — short instruction Q&A.

Each row is {instruction, output}. We treat `output` as the verbose answer and caveman-rewrite.
Filter: English-only, code-heavy or substantive (>= 40 words output), not too long (<= 800
words to keep within context).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from data.sources._util import cli_main, looks_english, word_count


def iter_records(limit: int = 1500) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= limit:
            return
        instr = (row.get("instruction") or "").strip()
        out = (row.get("output") or "").strip()
        if not instr or not out:
            continue
        wc = word_count(out)
        if wc < 40 or wc > 800:
            continue
        if not looks_english(instr) or not looks_english(out):
            continue
        emitted += 1
        yield {
            "prompt": instr,
            "source_normal": out,
            "source_seed": None,
            "category": "qa",
            "origin": f"evol-codealpaca:{emitted}",
            "license": "apache-2.0",
        }


if __name__ == "__main__":
    cli_main("evol_codealpaca", iter_records)
