"""HuggingFaceH4/ultrachat_200k — short technical SFT turns.

Each row is a multi-turn `messages` list. We take only the FIRST user/assistant pair as
short Q&A (the dialogue bucket is handled by oasst2). Filter: technical-ish (mentions code,
debug, error, function, etc.) and length-bounded.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from data.sources._util import cli_main, looks_english, word_count

TECH_RE = re.compile(
    r"\b(code|function|class|method|api|database|query|debug|error|exception|"
    r"bug|stack ?trace|compile|deploy|server|client|library|framework|"
    r"algorithm|http|json|xml|sql|python|javascript|typescript|rust|golang|java|"
    r"c\+\+|kotlin|swift|docker|kubernetes|git)\b",
    re.IGNORECASE,
)


def iter_records(limit: int = 500) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= limit:
            return
        msgs = row.get("messages") or []
        if len(msgs) < 2:
            continue
        user = next((m for m in msgs if m.get("role") == "user"), None)
        asst = next((m for m in msgs if m.get("role") == "assistant"), None)
        if not user or not asst:
            continue
        q = (user.get("content") or "").strip()
        a = (asst.get("content") or "").strip()
        if not q or not a:
            continue
        wc = word_count(a)
        if wc < 50 or wc > 600:
            continue
        if not (TECH_RE.search(q) or TECH_RE.search(a)):
            continue
        if not looks_english(q) or not looks_english(a):
            continue
        emitted += 1
        yield {
            "prompt": q,
            "source_normal": a,
            "source_seed": None,
            "category": "qa",
            "origin": f"ultrachat:{row.get('prompt_id', emitted)}",
            "license": "mit",
        }


if __name__ == "__main__":
    cli_main("ultrachat", iter_records)
