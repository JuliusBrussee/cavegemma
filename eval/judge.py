"""LLM-as-judge: score 20 holdouts on rule-compliance + code-preservation.

Uses `claude` CLI in headless mode. Each judgment returns a JSON like:
    {"rule_compliance": 1-5, "code_preservation": 1-5, "compression": 1-5, "comment": "..."}

Aggregates to a single mean; target >=4.0.

Run:
    python eval/judge.py --in artifacts/eval_predictions.jsonl --n 20 --out artifacts/judge.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = ROOT / "data" / "seeds" / "skill_md.md"


def judge_system() -> str:
    ruleset = SKILL_MD.read_text(encoding="utf-8") if SKILL_MD.exists() else "(SKILL.md missing)"
    return (
        "You are a strict evaluator for caveman-mode rewrites. Score on three axes 1-5 each:\n"
        "  rule_compliance  — drops articles/filler/pleasantries, fragments fine, terse\n"
        "  code_preservation — code blocks, identifiers, error strings, CLI commands byte-exact\n"
        "  compression       — meaningfully shorter than source while preserving full meaning\n"
        "Output ONLY a single JSON object on one line with keys rule_compliance, code_preservation, "
        "compression, comment. No markdown fences. Comment <=20 words.\n\n"
        "=== RULESET ===\n" + ruleset
    )


def run_judge(source: str, prediction: str, model: str) -> dict:
    user = (
        "SOURCE (original):\n" + source +
        "\n\n===\n\nREWRITE (to judge):\n" + prediction +
        "\n\nReturn JSON only."
    )
    cmd = [
        "claude", "-p", user,
        "--model", model,
        "--append-system-prompt", judge_system(),
        "--output-format", "json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:300])
    obj = json.loads(proc.stdout)
    raw = obj.get("result", "").strip()
    # Strip code-fence wrapping if model added one despite instructions.
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1].rstrip("`")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"judge returned non-JSON: {raw[:200]}") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="claude-opus-4-7")
    args = ap.parse_args()

    if not shutil.which("claude"):
        sys.exit("`claude` CLI not on PATH.")

    rows = [json.loads(ln) for ln in args.inp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    judgments: list[dict] = []
    with args.out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(sample):
            try:
                j = run_judge(r["source"], r["prediction"], args.model)
            except Exception as e:
                j = {"error": str(e)}
            rec = {"i": i, "prompt": r.get("prompt"), **j}
            judgments.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {i+1}/{len(sample)}  {j}")

    # Aggregate
    axes = ["rule_compliance", "code_preservation", "compression"]
    print("\n--- judge summary ---")
    overall = []
    for axis in axes:
        vals = [j[axis] for j in judgments if isinstance(j.get(axis), (int, float))]
        if not vals:
            print(f"  {axis:20s} no scores")
            continue
        mean = sum(vals) / len(vals)
        overall.append(mean)
        print(f"  {axis:20s} mean={mean:.2f}  n={len(vals)}")
    if overall:
        print(f"  {'OVERALL':20s} mean={sum(overall)/len(overall):.2f}  (target >= 4.0)")


if __name__ == "__main__":
    main()
