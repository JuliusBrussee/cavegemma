"""Fine-tune Gemma 4 31B with QLoRA via Unsloth + TRL SFTTrainer.

Designed to run on a single RunPod H100 80GB PCIe pod.
All hyperparams live in training/config.toml.

Run:
    python training/train_unsloth.py --config training/config.toml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must be set BEFORE `import unsloth`. Required by Unsloth ≥ 2024.11 when used with
# TRL's SFTTrainer — otherwise compute_loss raises NotImplementedError.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

try:
    import tomllib  # py311+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def build_messages(row: dict, instruction: str) -> list[dict]:
    """Pack a {prompt?, source, target} row into a Gemma chat-template messages list."""
    user = f"{instruction}\n\n{row['source']}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["target"]},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mc, lc, dc, tc = cfg["model"], cfg["lora"], cfg["data"], cfg["train"]

    # Lazy imports — only meaningful on a GPU host with all deps installed.
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    print(f">>> Loading base model: {mc['base']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mc["base"],
        max_seq_length=mc["max_seq_length"],
        load_in_4bit=mc["load_in_4bit"],
        dtype=None,  # let Unsloth pick bf16 on H100
    )
    # Gemma 4 is multimodal; Unsloth returns a Gemma4Processor. TRL's SFTTrainer
    # needs the inner text tokenizer (with `convert_tokens_to_ids`, padding, etc).
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    print(">>> Attaching LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        bias=lc["bias"],
        use_gradient_checkpointing=lc["use_gradient_checkpointing"],
        use_rslora=lc["use_rslora"],
    )

    print(">>> Loading dataset")
    raw = load_dataset(
        "json",
        data_files={"train": dc["train_path"], "eval": dc["eval_path"]},
    )

    instruction: str = dc["instruction"]

    def to_chat(row):
        msgs = build_messages(row, instruction)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = raw.map(to_chat, remove_columns=raw["train"].column_names)
    print(f"  train={len(dataset['train'])}  eval={len(dataset['eval'])}")
    print("--- sample train example ---")
    print(dataset["train"][0]["text"][:1200])
    print("--- end sample ---")

    sft_cfg = SFTConfig(
        output_dir=tc["output_dir"],
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        optim=tc["optim"],
        bf16=tc["bf16"],
        logging_steps=tc["logging_steps"],
        save_strategy=tc["save_strategy"],
        eval_strategy=tc["eval_strategy"],
        save_total_limit=tc["save_total_limit"],
        report_to=tc["report_to"],
        seed=tc["seed"],
        packing=tc["packing"],
        max_seq_length=mc["max_seq_length"],
        completion_only_loss=True,   # mask user turn from loss (TRL 0.17 API)
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,   # renamed from `tokenizer` in TRL 0.17
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=sft_cfg,
    )

    print(">>> Starting training")
    # Auto-resume from the latest checkpoint in output_dir if one exists.
    out_dir = Path(tc["output_dir"])
    resume = any(p.name.startswith("checkpoint-") for p in out_dir.glob("*")) if out_dir.exists() else False
    if resume:
        print(f"    found existing checkpoint(s) under {out_dir} — resuming")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"   # belt-and-suspenders before train()
    trainer.train(resume_from_checkpoint=resume)

    out = Path(tc["output_dir"])
    print(f">>> Saving final adapter -> {out}")
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    # Mini eval-loss summary so the pod logs leave a clear record.
    eval_metrics = trainer.evaluate()
    (out / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    print(json.dumps(eval_metrics, indent=2))
    print(">>> Done.")


if __name__ == "__main__":
    sys.exit(main())
