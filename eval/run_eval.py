"""Generate caveman rewrites for the eval set + workflow prompts, score against gates.

Run on RunPod (or any GPU box):
    python eval/run_eval.py \
        --adapter artifacts/adapter \
        --eval data/out/eval.jsonl \
        --workflow eval/workflow_prompts.jsonl \
        --out artifacts/eval_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `python eval/run_eval.py` from project root

from eval.metrics import aggregate  # noqa: E402

INSTRUCTION = (
    "Rewrite in caveman-mode. Drop articles, filler, pleasantries. "
    "Keep code blocks byte-exact. Preserve technical meaning."
)

WORKFLOW_INSTRUCTION = (
    "Answer in caveman-mode. Drop articles, filler, pleasantries. "
    "Keep code blocks byte-exact. Preserve full technical meaning."
)

# Per-category gates. workflows compress less, so widen the compression band.
GATES = {
    "qa":       {"compression": (0.30, 0.65), "article_density_target": (None, 0.02),
                 "code_fence_match": (0.95, None), "semantic_sim": (0.75, None)},
    "review":   {"compression": (0.40, 0.75), "article_density_target": (None, 0.02),
                 "code_fence_match": (0.95, None), "semantic_sim": (0.70, None)},
    "debug":    {"compression": (0.40, 0.75), "article_density_target": (None, 0.02),
                 "code_fence_match": (0.95, None), "semantic_sim": (0.70, None)},
    "refactor": {"compression": (0.40, 0.75), "article_density_target": (None, 0.02),
                 "code_fence_match": (0.95, None), "semantic_sim": (0.70, None)},
    "dialogue": {"compression": (0.40, 0.80), "article_density_target": (None, 0.02),
                 "code_fence_match": (0.95, None), "semantic_sim": (0.65, None)},
}


def _check(metric: str, mean: float, gate: tuple) -> bool:
    lo, hi = gate
    if lo is not None and mean < lo:
        return False
    return not (hi is not None and mean > hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base", default="google/gemma-4-31B-it")
    ap.add_argument("--eval", dest="ev", required=True, type=Path,
                    help="JSONL holdout from data/split.py")
    ap.add_argument("--workflow", type=Path, default=ROOT / "eval" / "workflow_prompts.jsonl",
                    help="hand-curated workflow prompts (no reference; eval prompt-only)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel

    print(f">>> Loading base {args.base} + adapter {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)   # unwrap Gemma4Processor
    FastLanguageModel.for_inference(model)

    def generate(user_msg: str) -> str:
        msgs = [{"role": "user", "content": user_msg}]
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        out = model.generate(
            input_ids=ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    # --- holdout eval (has references)
    eval_rows = [json.loads(ln) for ln in args.ev.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    print(f">>> Generating {len(eval_rows)} holdout predictions")

    preds: list[dict] = []
    pairs_by_cat: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for i, row in enumerate(eval_rows):
        gen = generate(f"{INSTRUCTION}\n\n{row['source']}")
        category = row.get("category", "qa")
        rec = {
            "split": "holdout",
            "category": category,
            "prompt": row.get("prompt"),
            "source": row["source"],
            "reference": row["target"],
            "prediction": gen,
        }
        preds.append(rec)
        pairs_by_cat[category].append((row["source"], gen))
        if (i + 1) % 10 == 0:
            print(f"  holdout {i+1}/{len(eval_rows)}")

    # --- workflow eval (no reference, prompt-only)
    workflow_rows = (
        [json.loads(ln) for ln in args.workflow.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if args.workflow.exists()
        else []
    )
    print(f">>> Generating {len(workflow_rows)} workflow predictions")
    for i, row in enumerate(workflow_rows):
        gen = generate(f"{WORKFLOW_INSTRUCTION}\n\n{row['prompt']}")
        category = row.get("category", "qa")
        rec = {
            "split": "workflow",
            "category": category,
            "prompt": row["prompt"],
            "source": row["prompt"],   # no separate verbose source here
            "reference": None,
            "prediction": gen,
        }
        preds.append(rec)
        # For workflow prompts we don't have a verbose reference; we still measure
        # article-density and code-fence preservation. Compression vs source-prompt is
        # not meaningful, so we exclude these from compression aggregation by tagging.
        pairs_by_cat[f"workflow:{category}"].append((row["prompt"], gen))
        if (i + 1) % 5 == 0:
            print(f"  workflow {i+1}/{len(workflow_rows)}")

    with args.out.open("w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- score per category
    print("\n>>> Per-category summary")
    fails = 0
    for cat, pairs in pairs_by_cat.items():
        summary = aggregate(pairs)["aggregate"]
        base_cat = cat.split(":", 1)[-1]
        gates = GATES.get(base_cat, GATES["qa"])
        print(f"\n--- category={cat}  n={len(pairs)} ---")
        for metric in ("compression", "article_density_target", "code_fence_match", "semantic_sim"):
            if metric not in summary:
                continue
            # Skip compression gate for workflow:* (no verbose reference)
            if cat.startswith("workflow:") and metric == "compression":
                print(f"  {metric:24s} mean={summary[metric]['mean']:.3f}  (info-only, no reference)")
                continue
            mean = summary[metric]["mean"]
            gate = gates.get(metric)
            ok = _check(metric, mean, gate) if gate else True
            print(f"  {metric:24s} mean={mean:.3f}  gate={gate}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                fails += 1
    print(f"\n{fails} gate(s) failed." if fails else "\nall gates PASS.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
