"""bigcode/commitpackft — commit message + before/after code -> refactor walkthrough seed.

We bypass commitpackft's dead loader script (`datasets` v4+ doesn't allow scripts) by
loading the per-language jsonl shards directly via HTTPS. Only common languages we care
about; only permissively-licensed rows. Synthesizer NORMALIZES the seed into a refactor
narrative, then caveman-rewrites.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from data.sources._util import cli_main, looks_english, word_count

# Languages most likely to produce useful refactor narratives.
LANGS = (
    "python", "javascript", "typescript", "go", "rust", "java",
    "ruby", "c#", "kotlin", "swift",
)

PERMISSIVE = {
    "mit", "apache-2.0", "apache 2.0", "bsd-3-clause", "bsd-2-clause", "isc",
    "unlicense", "0bsd",
}

BASE_URL = "https://huggingface.co/datasets/bigcode/commitpackft/resolve/main/data/{lang}/data.jsonl"


def iter_records(limit: int = 400) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    data_files = [BASE_URL.format(lang=lang) for lang in LANGS]
    ds = load_dataset("json", data_files=data_files, split="train", streaming=True)

    emitted = 0
    for row in ds:
        if emitted >= limit:
            return
        license_ = (row.get("license") or "").strip().lower()
        if license_ not in PERMISSIVE:
            continue
        message = (row.get("message") or row.get("subject") or "").strip()
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

        def _cap(s: str, n: int = 1500) -> str:
            return s if len(s) < n else s[:n] + "\n... [truncated]"

        lang = (row.get("lang") or "").strip()
        seed = (
            f"COMMIT MESSAGE: {message}\n\n"
            f"BEFORE ({lang}):\n```{lang.lower()}\n{_cap(old_contents)}\n```\n\n"
            f"AFTER ({lang}):\n```{lang.lower()}\n{_cap(new_contents)}\n```"
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
