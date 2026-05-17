"""Quick smoke-test for a trained adapter against the 10 caveman repo eval prompts.

Run on RunPod or any GPU box:
    python scripts/infer.py --adapter artifacts/adapter
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "seeds" / "eval_prompts_en.txt"

INSTRUCTION = (
    "Rewrite in caveman-mode. Drop articles, filler, pleasantries. "
    "Keep code blocks byte-exact. Preserve technical meaning."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base", default="google/gemma-4-31B-it")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)   # unwrap Gemma4Processor
    FastLanguageModel.for_inference(model)

    prompts = [ln.strip() for ln in PROMPTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for p in prompts:
        msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{p}"}]
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
        gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        print("=" * 80)
        print("PROMPT:", p)
        print("-" * 80)
        print(gen)
        print()


if __name__ == "__main__":
    main()
