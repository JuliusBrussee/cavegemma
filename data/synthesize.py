"""Two-step synthesis: NORMALIZE (if needed) -> CAVEMAN rewrite.

Input row shape (from data/build_corpus.py):
    {prompt, source_normal?, source_seed?, category, origin, license}

If `source_normal` is present (most sources): one Claude call — caveman-rewrite source_normal.
If only `source_seed` is present (SWE-bench, commitpack): two calls — normalize seed into
verbose workflow narrative, then caveman-rewrite that narrative.

Output row shape (raw_pairs.jsonl):
    {key, prompt, source, target, origin, category, license}

Drives `claude -p` non-interactive. Resumes on rerun via key-hash dedup.

Run:
    python data/synthesize.py --in data/out/corpus_raw.jsonl --out data/out/raw_pairs.jsonl
    python data/synthesize.py --limit 20    # spot check
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = ROOT / "data" / "seeds" / "skill_md.md"

CATEGORY_NORMALIZE_SYSTEM = {
    "debug": (
        "You are a senior engineer writing a debug-session walkthrough for a peer. "
        "Given a GitHub issue + the patch that fixed it, write a verbose multi-step debug narrative: "
        "1) reproduce, 2) hypothesize, 3) instrument/inspect, 4) fix, 5) verify. "
        "Quote relevant code from the patch in fenced blocks. 200-450 words. Markdown."
    ),
    "refactor": (
        "You are a senior engineer documenting a refactor for code review. "
        "Given a commit message + before/after code, write a verbose walkthrough: "
        "why this changed, what specifically moved, risk surface, how to verify. "
        "Quote before/after snippets in fenced blocks. 150-350 words. Markdown."
    ),
    "review": (
        "You are a senior engineer writing detailed code-review feedback. "
        "Given a diff hunk + a real review comment, expand into a full review note: "
        "what the issue is, why it matters, suggested fix with example code. "
        "150-300 words. Markdown."
    ),
    "dialogue": (  # rarely used — oasst2 already has source_normal
        "Rewrite this multi-turn technical dialogue verbosely, preserving every turn's intent "
        "and any code blocks byte-exact."
    ),
    "qa": (
        "You are a senior engineer answering a peer's question. Be helpful, complete, clear. "
        "Markdown with code blocks where appropriate. 80-300 words."
    ),
}

CAVEMAN_INSTRUCTIONS = (
    "You are a rewriting engine. Take the input answer and rewrite it in caveman-mode "
    "(level: full) per the rules below. CRITICAL invariants:\n"
    "  - All code blocks (between ``` fences) pass through BYTE-EXACT. Do NOT change "
    "identifiers, whitespace, or syntax inside code fences.\n"
    "  - All quoted error strings, API names, function names, CLI commands: exact.\n"
    "  - Drop articles (a/an/the), filler (just/really/basically/actually/simply), "
    "pleasantries, hedging. Fragments OK. Pattern: [thing] [action] [reason]. [next step].\n"
    "  - Preserve full technical meaning. Brain big, mouth small.\n"
    "  - If the input is a multi-turn dialogue (### user / ### assistant headings), "
    "rewrite ONLY the assistant turns in caveman; keep user turns and ### headings byte-exact.\n"
    "Return ONLY the rewritten answer. No preamble, no commentary, no markdown fence "
    "around the whole output.\n\n"
    "=== RULESET ===\n"
)


def caveman_system() -> str:
    skill = SKILL_MD.read_text(encoding="utf-8") if SKILL_MD.exists() else ""
    return CAVEMAN_INSTRUCTIONS + skill


def _hash_key(origin: str) -> str:
    return sha1(origin.encode("utf-8")).hexdigest()[:16]


def _have_claude() -> bool:
    return shutil.which("claude") is not None


def _have_codex() -> bool:
    return shutil.which("codex") is not None


def _run_codex(system: str, user: str, model: str, timeout_s: int = 300) -> str:
    """codex exec has no separate system-prompt flag, so we fold instructions + input
    into a single stdin payload. `--output-last-message` extracts just the model's
    final reply (no session preamble or token counters)."""
    full_prompt = (
        "=== INSTRUCTIONS ===\n"
        + system
        + "\n\n=== INPUT ===\n"
        + user
        + "\n\n=== OUTPUT (ONLY the rewritten content, no preamble, no commentary, "
          "no markdown fence wrapping the whole output) ==="
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        out_path = f.name
    cmd = [
        "codex", "exec",
        "-m", model,
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-rules",
        "--ignore-user-config",
        "--color", "never",
        "--output-last-message", out_path,
    ]
    last_err = ""
    for attempt in range(3):
        try:
            proc = subprocess.run(
                cmd, input=full_prompt, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            last_err = f"timeout: {e}"
        else:
            if proc.returncode == 0:
                try:
                    with open(out_path, encoding="utf-8") as f:
                        out = f.read().strip()
                    if out:
                        return out
                    last_err = "codex empty output"
                except OSError as e:
                    last_err = f"codex output unreadable: {e}"
            else:
                last_err = f"codex exit {proc.returncode}: {proc.stderr.strip()[:400]}"
        time.sleep(2 + 4 * attempt + random.random())
    import contextlib
    with contextlib.suppress(OSError):
        os.unlink(out_path)
    raise RuntimeError(last_err)


def _run_claude(system: str, user: str, model: str, timeout_s: int = 300) -> str:
    # Pass `user` via stdin, NOT as a positional arg. User content can begin with
    # `---` (diff patches), which `claude -p ---xyz` parses as a CLI option.
    cmd = [
        "claude", "-p",
        "--model", model,
        "--append-system-prompt", system,
        "--output-format", "json",
    ]
    last_err = ""
    # Up to 3 attempts with backoff for transient rate-limit / concurrency hiccups.
    for attempt in range(3):
        try:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            last_err = f"timeout: {e}"
        else:
            if proc.returncode == 0:
                obj = json.loads(proc.stdout)
                if isinstance(obj, dict) and "result" in obj:
                    return obj["result"].strip()
                raise RuntimeError(f"unexpected claude json shape: {list(obj)[:5]}")
            last_err = f"claude exit {proc.returncode}: {proc.stderr.strip()[:400]}"
        # backoff: 2s, 6s
        time.sleep(2 + 4 * attempt + random.random())
    raise RuntimeError(last_err)


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            keys.add(json.loads(line)["key"])
        except Exception:
            continue
    return keys


def synth_one(row: dict, model_normalize: str, model_caveman: str, backend: str) -> dict:
    runner = _run_codex if backend == "codex" else _run_claude
    category = row.get("category", "qa")
    if row.get("source_normal"):
        normal = row["source_normal"]
    elif row.get("source_seed"):
        sys_p = CATEGORY_NORMALIZE_SYSTEM.get(category, CATEGORY_NORMALIZE_SYSTEM["qa"])
        normal = runner(sys_p, row["source_seed"], model_normalize)
    else:
        raise ValueError(f"row {row.get('origin')} has neither source_normal nor source_seed")
    caveman = runner(caveman_system(), normal, model_caveman)
    return {
        "key": _hash_key(row["origin"]),
        "prompt": row["prompt"],
        "source": normal,
        "target": caveman,
        "origin": row["origin"],
        "category": category,
        "license": row.get("license"),
        "backend": backend,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path,
                    default=ROOT / "data" / "out" / "corpus_raw.jsonl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "out" / "raw_pairs.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", choices=("claude", "codex"), default="claude")
    ap.add_argument("--model-normalize", default=None,
                    help="claude default: claude-sonnet-4-6; codex default: gpt-5.5")
    ap.add_argument("--model-caveman", default=None,
                    help="claude default: claude-opus-4-7; codex default: gpt-5.5")
    args = ap.parse_args()

    if args.backend == "claude" and not _have_claude():
        sys.exit("`claude` CLI not on PATH.")
    if args.backend == "codex" and not _have_codex():
        sys.exit("`codex` CLI not on PATH.")
    if SKILL_MD.exists() is False:
        sys.exit(f"missing {SKILL_MD}.")

    # Default models per backend.
    if args.model_normalize is None:
        args.model_normalize = "gpt-5.5" if args.backend == "codex" else "claude-sonnet-4-6"
    if args.model_caveman is None:
        args.model_caveman = "gpt-5.5" if args.backend == "codex" else "claude-opus-4-7"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_keys(args.out)

    rows: list[dict] = []
    for line in args.inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if _hash_key(rec["origin"]) in done:
            continue
        rows.append(rec)
    # Shuffle to interleave sources — keeps per-source failures from clustering and
    # triggering the consecutive-failure circuit breaker before other sources run.
    random.Random(42).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    print(f"synthesizing {len(rows)} new pairs (skipped {len(done)} done)")
    t0 = time.time()
    written = 0
    consecutive_errors = 0
    error_streak_limit = 25  # halt if 25 in a row fail (Claude down / usage hit)
    halted = False
    with args.out.open("a", encoding="utf-8") as f, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(synth_one, r, args.model_normalize, args.model_caveman, args.backend): r
            for r in rows
        }
        for fut in concurrent.futures.as_completed(futures):
            r = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  ! {r.get('origin','?')}: {e}", file=sys.stderr)
                consecutive_errors += 1
                if consecutive_errors >= error_streak_limit:
                    print(
                        f"  !! {consecutive_errors} consecutive failures — halting. "
                        f"Wrote {written} pairs. Rerun this script to resume.",
                        file=sys.stderr,
                    )
                    halted = True
                    for pending in futures:
                        pending.cancel()
                    break
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            consecutive_errors = 0
            if written % 20 == 0:
                rate = written / (time.time() - t0)
                print(f"  {written}/{len(rows)} @ {rate:.2f}/s")
    print(f"done. wrote {written} pairs in {time.time()-t0:.1f}s.")
    if halted:
        sys.exit(2)


if __name__ == "__main__":
    main()
