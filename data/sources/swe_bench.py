"""princeton-nlp/SWE-bench_Verified — issue + patch -> debug-session seed.

We don't have a debug narrative in this dataset, only (issue_body, patch). The synthesizer
will run the NORMALIZE step on these to produce a verbose debug-session narrative, then
caveman-rewrite it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from data.sources._util import cli_main, looks_english, word_count


def iter_records(limit: int = 500) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= limit:
            return
        problem = row.get("problem_statement", "") or ""
        patch = row.get("patch", "") or ""
        if not problem.strip() or not patch.strip():
            continue
        if not looks_english(problem):
            continue
        if word_count(problem) < 30:
            continue
        # Cap patch length to keep within model context.
        patch_trim = patch if len(patch) < 4000 else patch[:4000] + "\n... [truncated]"
        seed = (
            "ISSUE:\n"
            + problem.strip()
            + "\n\n---\n\nPATCH (the fix that landed):\n```diff\n"
            + patch_trim
            + "\n```"
        )
        repo = row.get("repo", "?")
        instance = row.get("instance_id", "?")
        emitted += 1
        yield {
            "prompt": f"Debug-session walkthrough for {repo} issue {instance}",
            "source_normal": None,
            "source_seed": seed,
            "category": "debug",
            "origin": f"swe-bench-verified:{instance}",
            "license": "mit",  # SWE-bench is permissive; verify before commercial use
        }


if __name__ == "__main__":
    cli_main("swe_bench", iter_records)
