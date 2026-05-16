"""Caveman-fidelity metrics. Pure-Python where possible; optional MiniLM for semantic sim.

Metrics + gates (calibrated against caveman repo's gold __baseline__ -> caveman pairs):
  - compression_ratio = tok(target) / tok(source)     gate 0.30-0.65  (gold mean 0.54)
  - article_density(target)                           gate <=0.02     (gold 0.003)
  - code_fence_exact_match(source, target)            gate >=0.95     (gold 0.90; one outlier drops decorative code)
  - semantic_sim(source, target)                      gate >=0.75     (gold 0.79)

The trained model is allowed to be stricter than gold (e.g. >0.95 fence match) but the
gates must not exceed what gold itself achieves, or we will fail eval on our own training
distribution.

CLI self-test uses the caveman repo's results.json to confirm metrics are calibrated:
    python eval/metrics.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data" / "seeds" / "caveman_results.json"

FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-.]*\n?(.*?)```", re.DOTALL)
ARTICLE_RE = re.compile(r"\b(the|a|an|is|are|was|were)\b", re.IGNORECASE)
WORD_RE = re.compile(r"\b\w+\b")


def tok_count(text: str) -> int:
    # tiktoken is more accurate; fall back to simple word count if not available.
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(WORD_RE.findall(text))


def compression_ratio(source: str, target: str) -> float:
    s = tok_count(source)
    return tok_count(target) / s if s else 0.0


def article_density(text: str) -> float:
    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(1 for _ in ARTICLE_RE.finditer(text)) / len(words)


def extract_fences(text: str) -> list[str]:
    return [m.group(1).rstrip("\n") for m in FENCE_RE.finditer(text)]


def code_fence_exact_match(source: str, target: str) -> float:
    fences = [f for f in extract_fences(source) if f.strip()]
    if not fences:
        return 1.0
    return sum(1 for f in fences if f in target) / len(fences)


_ST_MODEL = None


def _semantic_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError("install sentence-transformers for semantic sim") from e
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST_MODEL


def semantic_sim(source: str, target: str) -> float:
    m = _semantic_model()
    embs = m.encode([source, target], convert_to_numpy=True, normalize_embeddings=True)
    return float((embs[0] * embs[1]).sum())


def score_pair(source: str, target: str, *, with_semantic: bool = True) -> dict:
    out = {
        "compression": compression_ratio(source, target),
        "article_density_target": article_density(target),
        "code_fence_match": code_fence_exact_match(source, target),
    }
    if with_semantic:
        try:
            out["semantic_sim"] = semantic_sim(source, target)
        except Exception as e:
            out["semantic_sim"] = None
            out["semantic_sim_error"] = str(e)
    return out


def aggregate(pairs: list[tuple[str, str]], *, with_semantic: bool = True) -> dict:
    scores = [score_pair(s, t, with_semantic=with_semantic) for s, t in pairs]
    keys = ["compression", "article_density_target", "code_fence_match"]
    if with_semantic:
        keys.append("semantic_sim")
    agg = {}
    for k in keys:
        vals = [s[k] for s in scores if isinstance(s.get(k), (int, float))]
        if not vals:
            continue
        agg[k] = {
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return {"per_pair": scores, "aggregate": agg}


def _self_test() -> None:
    """Score caveman repo's own baseline->caveman pairs. Should land in spec ranges."""
    if not SEEDS.exists():
        sys.exit(f"missing {SEEDS}. run the seed-fetch step.")
    data = json.loads(SEEDS.read_text(encoding="utf-8"))
    base = data["arms"]["__baseline__"]
    cave = data["arms"]["caveman"]
    pairs = list(zip(base, cave, strict=False))
    result = aggregate(pairs)
    print(json.dumps(result["aggregate"], indent=2))
    agg = result["aggregate"]
    print()
    print("--- gate check (against caveman repo gold pairs) ---")
    cr = agg["compression"]["mean"]
    ad = agg["article_density_target"]["mean"]
    cf = agg["code_fence_match"]["mean"]
    print(f"compression mean   : {cr:.3f}   (gate 0.30-0.65)  {'PASS' if 0.30 <= cr <= 0.65 else 'OUT'}")
    print(f"article density    : {ad:.3f}   (gate <= 0.02)    {'PASS' if ad <= 0.02 else 'OUT'}")
    print(f"code fence match   : {cf:.3f}   (gate >= 0.95)    {'PASS' if cf >= 0.95 else 'OUT'}")
    if "semantic_sim" in agg:
        ss = agg["semantic_sim"]["mean"]
        print(f"semantic sim       : {ss:.3f}   (gate >= 0.75)    {'PASS' if ss >= 0.75 else 'OUT'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--in", dest="inp", type=Path, help="JSONL with source/target keys")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.inp:
        ap.error("provide --self-test or --in PATH")
    pairs = []
    for line in args.inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pairs.append((r["source"], r["target"]))
    print(json.dumps(aggregate(pairs)["aggregate"], indent=2))


if __name__ == "__main__":
    main()
