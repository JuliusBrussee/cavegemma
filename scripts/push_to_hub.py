"""Push the LoRA adapter + a model card to HuggingFace Hub.

Run:
    huggingface-cli login
    python scripts/push_to_hub.py --adapter artifacts/adapter --repo julb/gemma-4-31B-caveman-lora
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

MODEL_CARD = dedent(
    """\
    ---
    base_model: google/gemma-4-31B-it
    library_name: peft
    license: gemma
    tags:
      - lora
      - caveman
      - style-transfer
      - gemma-license
    ---

    # gemma-4-31B-caveman-lora

    LoRA adapter that makes Gemma 4 31B speak "caveman-mode" — drops articles, filler, pleasantries; keeps technical accuracy; leaves code blocks byte-exact.

    Style source-of-truth: [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) (MIT).

    ## Usage

    ```python
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained("google/gemma-4-31B-it", load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    model = PeftModel.from_pretrained(base, "{repo}")

    msgs = [{{"role": "user", "content": "Rewrite in caveman-mode. Drop articles, filler, pleasantries. Keep code blocks byte-exact.\\n\\nWhy does my React component re-render?"}}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(ids, max_new_tokens=300, do_sample=False)
    print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    ```

    ## Training

    - QLoRA NF4 + double-quant + bf16 compute, rank 16, alpha 32
    - TRL `SFTTrainer` with `assistant_only_loss=True`
    - 3 epochs, lr 2e-4 cosine, batch 4 × grad accum 4 (effective 16)
    - Single RunPod H100 80GB PCIe

    Inherits the **Gemma Prohibited Use Policy** from the base model.
    """
).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--repo", required=True, help="hf repo id, e.g. julb/gemma-4-31B-caveman-lora")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo

    card = MODEL_CARD.replace("{repo}", args.repo)
    card_path = args.adapter / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f">>> wrote {card_path}")

    create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    api = HfApi()
    api.upload_folder(
        folder_path=str(args.adapter),
        repo_id=args.repo,
        repo_type="model",
        commit_message="upload caveman LoRA adapter",
    )
    print(f">>> pushed -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
