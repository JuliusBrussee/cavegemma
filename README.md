# finetune-caveman

LoRA fine-tune of **Gemma 4 31B Dense** to speak caveman-mode natively. Source of truth for behavior: [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) (MIT).

Output: HuggingFace adapter loadable on top of `google/gemma-4-31B-it`.

## Stack

- **Base**: `google/gemma-4-31B-it`
- **Method**: QLoRA NF4 + double-quant + bf16 compute (Unsloth)
- **Hardware**: RunPod H100 80GB PCIe community (~$1.99/hr, ~$3–5 total)
- **Loss**: assistant-only + completion-only masking via TRL `SFTConfig`
- **Eval gates**: compression ratio 0.30–0.50, article-drop ≤0.02, code-fence exact-match ≥0.99, MiniLM semantic-sim ≥0.75, LLM judge mean ≥4.0

## Layout

```
data/
  seeds/           caveman repo snapshots (results.json, SKILL.md, prompts)
  prompts/         tech-Q&A prompt corpus (input to synthesis)
  synthesize.py    drives Claude Code subagents → {normal, caveman} pairs
  filter.py        code-fence integrity + dedup + sanity filters
  split.py         90/10 train/eval split
  out/             train.jsonl, eval.jsonl, rejected.jsonl
training/
  train_unsloth.py runs on RunPod
  runpod_bootstrap.sh
  config.toml
eval/
  metrics.py       compression / article-drop / code-fence / semantic
  run_eval.py      score adapter on holdout
  judge.py         LLM-judge on 20 holdouts
scripts/
  infer.py
  push_to_hub.py
```

## Pipeline

```
prompts/tech_qa_seed.jsonl
  └─► synthesize.py ──► raw_pairs.jsonl
        └─► filter.py ──► clean_pairs.jsonl
              └─► split.py ──► train.jsonl + eval.jsonl
                    └─► (rsync to RunPod) ──► train_unsloth.py
                          └─► adapter checkpoint
                                └─► run_eval.py (gates)
                                      └─► push_to_hub.py
```

## Reproduce

```bash
# Local — data prep
uv sync
python data/build_prompts.py        # ~3-5k tech prompts
python data/synthesize.py           # Claude Code synthesis
python data/filter.py
python data/split.py

# RunPod — training
bash training/runpod_bootstrap.sh
python training/train_unsloth.py --config training/config.toml

# Eval
python eval/run_eval.py --adapter artifacts/adapter
python eval/judge.py

# Ship
python scripts/push_to_hub.py --adapter artifacts/adapter --repo julb/gemma-4-31B-caveman-lora
```

## License

Code: MIT. Adapter inherits Gemma Prohibited Use Policy from base model.
