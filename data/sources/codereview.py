"""ronantakizawa/github-codereview — diff + review-comment pairs with severity tags.

Each row holds before/after code, a `diff_context` hunk, and a human `reviewer_comment`
tagged with `comment_type` (bug/security/performance/refactor/style/suggestion/question).
We feed the diff as context and the human comment as source_normal (verbose review
feedback). Caveman-rewrite produces terse [line:] severity: problem. fix. format that
matches caveman repo's caveman-review.toml.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from data.sources._util import cli_main, looks_english, word_count

# All comment types are legitimate review styles; we filter only on quality + length.
KEEP_TYPES: set[str] = set()  # empty = accept all


def iter_records(limit: int = 500) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("ronantakizawa/github-codereview", split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= limit:
            return
        comment = (row.get("reviewer_comment") or "").strip()
        diff = (row.get("diff_context") or "").strip()
        ctype = (row.get("comment_type") or "").lower()
        quality = row.get("quality_score") or 0
        if not comment or not diff:
            continue
        if KEEP_TYPES and ctype not in KEEP_TYPES:
            continue
        if word_count(comment) < 6 or word_count(comment) > 500:
            continue
        if quality and quality < 0.2:
            continue
        if not looks_english(comment):
            continue
        diff_trim = diff if len(diff) < 2500 else diff[:2500] + "\n... [truncated]"
        lang = (row.get("language") or "").lower()
        prompt = f"Review this {ctype or 'change'} and explain the issue + fix."
        normal = (
            f"DIFF ({lang}):\n```{lang}\n"
            + diff_trim
            + "\n```\n\nREVIEW COMMENT:\n"
            + comment.strip()
        )
        idx = row.get("pr_number") or emitted
        emitted += 1
        yield {
            "prompt": prompt,
            "source_normal": normal,
            "source_seed": None,
            "category": "review",
            "origin": f"github-codereview:{idx}:{ctype or 'unspec'}:{emitted}",
            "license": "mit",
        }


if __name__ == "__main__":
    cli_main("codereview", iter_records)
