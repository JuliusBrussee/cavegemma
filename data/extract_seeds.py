"""Extract aligned {normal, caveman} pairs from caveman repo's results.json snapshot.

Output: data/out/seed_pairs.jsonl with 20 pairs (10 baseline->caveman, 10 terse->caveman).
These are gold-quality, human-authored exemplars and should be weighted heavily during training.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data" / "seeds" / "caveman_results.json"
OUT = ROOT / "data" / "out" / "seed_pairs.jsonl"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(SEEDS.read_text(encoding="utf-8"))
    prompts = data["prompts"]
    arms = data["arms"]
    baseline = arms["__baseline__"]
    terse = arms["__terse__"]
    caveman = arms["caveman"]

    pairs: list[dict] = []
    for i, prompt in enumerate(prompts):
        # baseline (verbose) -> caveman: strongest contrast, primary signal
        pairs.append(
            {
                "prompt": prompt,
                "source": baseline[i],
                "target": caveman[i],
                "origin": "seed:baseline",
            }
        )
        # terse -> caveman: teaches "go further than just be brief"
        pairs.append(
            {
                "prompt": prompt,
                "source": terse[i],
                "target": caveman[i],
                "origin": "seed:terse",
            }
        )

    with OUT.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} seed pairs -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
