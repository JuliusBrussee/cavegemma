"""ronantakizawa/github-codereview — diff + review-comment pairs with severity tags.

Each row has a diff hunk and a human review comment plus `comment_type` (bug/security/
performance/refactor/style/suggestion/nitpick/question). We feed the diff as context and
the human comment as source_normal (verbose review feedback), and caveman-rewrite to terse
[line:] severity: problem. fix. format that matches caveman repo's caveman-review.toml.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from data.sources._util import cli_main, looks_english, word_count

KEEP_TYPES = {"bug", "security", "performance", "refactor", "suggestion"}


def iter_records(limit: int = 500) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("ronantakizawa/github-codereview", split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= limit:
            return
        comment = row.get("comment", "") or ""
        diff = row.get("diff_hunk", "") or row.get("hunk", "") or row.get("code", "") or ""
        ctype = (row.get("comment_type") or "").lower()
        if ctype and ctype not in KEEP_TYPES:
            continue
        if word_count(comment) < 15 or word_count(comment) > 400:
            continue
        if not diff.strip() or not looks_english(comment):
            continue
        diff_trim = diff if len(diff) < 2500 else diff[:2500] + "\n... [truncated]"
        prompt = f"Review this {ctype or 'change'} and explain the issue + fix."
        normal = (
            "DIFF:\n```diff\n"
            + diff_trim
            + "\n```\n\nREVIEW COMMENT:\n"
            + comment.strip()
        )
        idx = row.get("id") or row.get("comment_id") or emitted
        emitted += 1
        yield {
            "prompt": prompt,
            "source_normal": normal,
            "source_seed": None,
            "category": "review",
            "origin": f"github-codereview:{idx}:{ctype or 'unspec'}",
            "license": "mit",  # subset is permissive; per-row license not exposed
        }


if __name__ == "__main__":
    cli_main("codereview", iter_records)
