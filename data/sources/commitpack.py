"""bigcode/commitpackft — commit message + before/after diff -> refactor walkthrough seed.

Only Apache-2.0 / MIT / BSD subset rows are retained for licensing safety. We use the commit
message as a hint of intent and the diff as the change. The synthesizer NORMALIZES into a
refactor narrative (why -> what changed -> verify), then caveman-rewrites.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from data.sources._util import cli_main, looks_english, word_count

PERMISSIVE = {"mit", "apache-2.0", "apache 2.0", "bsd-3-clause", "bsd-2-clause", "isc"}


def iter_records(limit: int = 400) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    # Use the small, well-curated 'commitpackft' (filtered subset, per-row license).
    # Stream — 700k rows; we only need a few hundred.
    ds = load_dataset("bigcode/commitpackft", split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= limit:
            return
        license_ = (row.get("license") or "").strip().lower()
        if license_ not in PERMISSIVE:
            continue
        message = (row.get("message") or "").strip()
        old_contents = (row.get("old_contents") or "").strip()
        new_contents = (row.get("new_contents") or "").strip()
        if not message or not old_contents or not new_contents:
            continue
        if word_count(message) < 5 or word_count(message) > 150:
            continue
        if not looks_english(message):
            continue
        if old_contents == new_contents:
            continue
        # Cap snippet sizes to fit context.
        def _cap(s: str, n: int = 1500) -> str:
            return s if len(s) < n else s[:n] + "\n... [truncated]"
        lang = row.get("lang") or row.get("language") or ""
        seed = (
            f"COMMIT MESSAGE: {message}\n\n"
            f"BEFORE ({lang}):\n```{lang}\n{_cap(old_contents)}\n```\n\n"
            f"AFTER ({lang}):\n```{lang}\n{_cap(new_contents)}\n```"
        )
        sha = row.get("commit") or row.get("sha") or emitted
        emitted += 1
        yield {
            "prompt": f"Walk me through this {lang or 'code'} refactor.",
            "source_normal": None,
            "source_seed": seed,
            "category": "refactor",
            "origin": f"commitpackft:{sha}",
            "license": license_,
        }


if __name__ == "__main__":
    cli_main("commitpack", iter_records)
