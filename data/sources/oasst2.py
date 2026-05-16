"""OpenAssistant/oasst2 — multi-turn dialogue extraction.

Walk dialogue trees and emit prompter -> assistant -> prompter ... chains of 3-6 turns,
English-only, technical or general. Each emitted record holds the FULL chain serialized as
the source_normal so the synthesizer can rewrite the whole multi-turn block in one pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from data.sources._util import cli_main, has_code, looks_english, word_count


def _build_chain(node: dict, by_id: dict[str, dict]) -> list[dict]:
    chain = []
    cur = node
    while cur is not None:
        chain.append(cur)
        cur = by_id.get(cur.get("parent_id")) if cur.get("parent_id") else None
    chain.reverse()
    return chain


def _serialize(chain: list[dict]) -> tuple[str, str]:
    """Return (prompt, dialogue_block). prompt = first prompter turn; block = full chain."""
    lines: list[str] = []
    for msg in chain:
        role = "user" if msg["role"] == "prompter" else "assistant"
        lines.append(f"### {role}\n{msg['text'].strip()}")
    return chain[0]["text"].strip(), "\n\n".join(lines)


def iter_records(limit: int = 1000) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)

    by_id: dict[str, dict] = {}
    # oasst2 is small enough to buffer the English subset in memory.
    for row in ds:
        if row.get("lang") != "en":
            continue
        by_id[row["message_id"]] = dict(row)

    seen_chains: set[str] = set()
    emitted = 0
    for _mid, row in by_id.items():
        if emitted >= limit:
            return
        # Anchor on assistant leaves of length >= 3 turns.
        if row["role"] != "assistant":
            continue
        # Find children to ensure this is a *leaf* — skip if any other msg's parent_id == mid.
        # (Cheap heuristic: rely on rank/score to prefer the top reply.)
        if (row.get("rank") or 0) != 0:
            continue
        chain = _build_chain(row, by_id)
        if not (3 <= len(chain) <= 6):
            continue
        # Need at least one assistant turn with code OR substantial technical content.
        if not any(has_code(m["text"]) or word_count(m["text"]) >= 80 for m in chain if m["role"] == "assistant"):
            continue
        # English filter on every turn
        if not all(looks_english(m["text"]) for m in chain):
            continue
        prompt, block = _serialize(chain)
        key = f"oasst2:{chain[-1]['message_id']}"
        if key in seen_chains:
            continue
        seen_chains.add(key)
        emitted += 1
        yield {
            "prompt": prompt,
            "source_normal": block,
            "source_seed": None,
            "category": "dialogue",
            "origin": key,
            "license": "apache-2.0",
        }


if __name__ == "__main__":
    cli_main("oasst2", iter_records)
