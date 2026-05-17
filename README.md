# finetune-caveman

Fine-tune of **Gemma 4 31B** to speak [caveman-mode](https://github.com/JuliusBrussee/caveman) natively. Drops articles/filler/pleasantries, keeps code/identifiers/error-strings byte-exact. Brain big, mouth small.

**Shipped models (Hugging Face):**

| Repo | Format | Size | What it is |
|---|---|---|---|
| [`JBrussee/gemma-4-31B-caveman`](https://huggingface.co/JBrussee/gemma-4-31B-caveman) | bf16 merged | 62.5 GB | Full Gemma 4 31B with caveman trained in. Load as-is. |
| [`JBrussee/gemma-4-31B-caveman-lora`](https://huggingface.co/JBrussee/gemma-4-31B-caveman-lora) | LoRA adapter | 534 MB | Loads on top of `google/gemma-4-31B-it`. |

## Quick start

### Full merged model (no extra setup)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("JBrussee/gemma-4-31B-caveman")
model = AutoModelForCausalLM.from_pretrained(
    "JBrussee/gemma-4-31B-caveman",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

msgs = [{"role": "user", "content": "Why does my React component re-render every time the parent updates?"}]
inputs = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(inputs, max_new_tokens=300, do_sample=False)
print(tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True))
```

### Adapter on base (lighter download)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31B-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
model = PeftModel.from_pretrained(base, "JBrussee/gemma-4-31B-caveman-lora")
```

## What the model does

It rewrites or answers technical questions in caveman style. Source-of-truth ruleset is the [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) skill (MIT).

Example (verbose input → caveman output):

> **In:** Because by default, React re-renders all children when a parent re-renders, regardless of whether their props changed. To prevent it, wrap the component in `React.memo`. Caveats: `memo` does a shallow prop comparison; new object/array/function props created inline in the parent break it — wrap those in `useCallback`/`useMemo`. Children passed via `children` are new elements each render, so `memo` won't help there. Context updates still re-render consumers regardless of `memo`.
>
> **Out:** Parent re-render → child re-render by default. Props "change" each render if inline obj/array/fn → new ref → child sees new props. Fix: wrap child `React.memo(Child)`, stabilize props with `useMemo`/`useCallback`, lift state. Check: if memo'd but still re-renders, inline prop is culprit.

## Training summary

| Field | Value |
|---|---|
| Base | `google/gemma-4-31B-it` |
| Method | QLoRA NF4 + double-quant + bf16 compute |
| LoRA | rank 16, α 32, dropout 0, targets all linear |
| Dataset | 1750 train + 193 eval (debug / review / refactor / dialogue / qa) |
| Epochs | 3, lr 2e-4 cosine, batch 2 × grad accum 8 (eff 16), `completion_only_loss=True` |
| Hardware | RunPod RTX PRO 6000 Blackwell 96GB, ~$1.89/hr |
| Time | ~50 min training (Unsloth + TRL 0.17) |
| Final loss | train 0.024 · eval 0.72 · eval acc 81.5% |

## Eval results (on 193-pair holdout)

Tagged by source category. `code_fence_match` = fraction of source code fences appearing byte-exact in target.

| Category | n | compression | article density | code_fence | semantic_sim |
|---|---|---|---|---|---|
| dialogue | 28 | 0.59 | 0.020 | 1.000 | 0.91 |
| debug | 34 | 0.92 | 0.009 | 0.995 | 0.98 |
| refactor | 27 | 0.92 | 0.005 | 0.963 | 0.98 |
| qa | 104 | 0.65 | 0.007 | 1.000 | 0.92 |

**What the numbers say:**
- ✅ Code preservation excellent (96-100% fence-exact)
- ✅ Article density well under target (0.5-2%)
- ✅ Semantic preservation strong (91-98%)
- ⚠️ Compression weaker than the aspirational gates (0.30-0.75) — model compresses ~10-40% rather than the 50-70% the gold pairs achieve. Training data's filter upper bound was 1.00; relax that and retrain for tighter compression.

## Repo layout

```
finetune-caveman/
├── data/
│   ├── seeds/                 caveman repo snapshots (SKILL.md, eval prompts)
│   ├── sources/               per-source HuggingFace loaders
│   ├── build_corpus.py        orchestrator (6 sources → corpus_raw.jsonl)
│   ├── synthesize.py          claude/codex CLI driver, two-step rewrite, resumable
│   ├── filter.py              fence-integrity + dedup + compression band
│   ├── split.py               90/10 split with seed-pair pinning
│   └── prompts/, out/         (out is gitignored)
├── training/
│   ├── train_unsloth.py       Unsloth + TRL SFT trainer, resume from checkpoint
│   ├── runpod_bootstrap.sh    pip + auth bootstrap for fresh pod
│   └── config.toml            single source of truth for hyperparams
├── eval/
│   ├── metrics.py             compression / article-drop / code-fence / semantic_sim
│   ├── run_eval.py            score adapter on holdout + workflow prompts
│   ├── judge.py               LLM-judge via claude CLI on 20 holdouts
│   └── workflow_prompts.jsonl 10 hand-curated workflow eval prompts
├── scripts/
│   ├── infer.py               smoke-test against caveman eval prompts
│   └── push_to_hub.py         publish adapter + model card
└── artifacts/                 (gitignored)
```

## Reproduce

End-to-end. ~6-8 hours plus pod time (~$4-5).

```bash
# 1. Local setup
uv sync
uv run python data/extract_seeds.py            # 20 gold seed pairs
uv run python data/build_corpus.py             # 3000 rows from 6 HF sources

# 2. Synthesis (Claude Code or Codex CLI required)
uv run python data/synthesize.py --backend claude --workers 3      # or
uv run python data/synthesize.py --backend codex --workers 3

# 3. Filter + split
uv run python data/filter.py --in data/out/raw_pairs.jsonl --out data/out/clean_pairs.jsonl
uv run python data/split.py --in data/out/clean_pairs.jsonl

# 4. RunPod H100 / RTX PRO 6000 — rsync, ssh, bootstrap, train
rsync -avz --exclude='.git' --exclude='.venv' -e "ssh -p <port> -i ~/.ssh/id_ed25519" ./ root@<pod>:/workspace/finetune-caveman/
ssh -i ~/.ssh/id_ed25519 -p <port> root@<pod> "
  export HF_TOKEN=...
  export WANDB_API_KEY=...
  cd /workspace/finetune-caveman
  bash training/runpod_bootstrap.sh
  python training/train_unsloth.py --config training/config.toml
"

# 5. Eval + ship
python eval/run_eval.py --adapter artifacts/adapter --eval data/out/eval.jsonl --workflow eval/workflow_prompts.jsonl --out artifacts/eval_predictions.jsonl
python scripts/push_to_hub.py --adapter artifacts/adapter --repo <hf-user>/gemma-4-31B-caveman-lora
```

## Datasets used (all permissively licensed)

| Source | License | Pulled | Used for |
|---|---|---|---|
| [`OpenAssistant/oasst2`](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache 2.0 | 400 | Multi-turn dialogue |
| [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | research-permissive | 400 | Debug-session narratives |
| [`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview) | permissive subset | 400 | Code review |
| [`bigcode/commitpackft`](https://huggingface.co/datasets/bigcode/commitpackft) | MIT/Apache subset | 300 | Refactor walkthroughs |
| [`theblackcat102/evol-codealpaca-v1`](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) | Apache 2.0 | 1200 | Short technical Q&A |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | 300 | Short Q&A overflow |

After filtering: 1750 train + 193 eval pairs. Pair caveman side synthesized via Claude Code (`claude -p`) and Codex CLI (`codex exec` with GPT-5.5), routed through the canonical [SKILL.md](https://github.com/JuliusBrussee/caveman/blob/main/skills/caveman/SKILL.md) ruleset.

## Limitations

- **Compression weaker than gold caveman** — model averages 0.6-0.9 vs gold's 0.3-0.5. The training filter accepted any output ≤ 1.0× source length; tightening to ≤ 0.7 and retraining would push the model harder.
- **`review` category sparse** — codex pairs often mutated diff fences, so the filter dropped most. Only ~8 review pairs in the eval set; review behavior is mostly extrapolated from debug/refactor neighbors.
- **Workflow eval gates** in `eval/run_eval.py` for the open-ended prompts (`workflow_prompts.jsonl`) are partly meaningless because there's no reference — `code_fence_match` checks input prompt's fences in the answer, `semantic_sim` compares answer to question. Treat those as info-only.
- **Multimodal capability** — Gemma 4 is multimodal (vision + audio). Fine-tune was text-only on the language head; vision/audio paths untouched, should still work but unverified.

## License

- Code in this repo: **MIT**
- Adapter and merged model inherit the **Gemma Prohibited Use Policy** (Apache 2.0 + Gemma terms). See [Google's Gemma terms](https://ai.google.dev/gemma/terms).
- Style ruleset and seed pairs from [`JuliusBrussee/caveman`](https://github.com/JuliusBrussee/caveman): MIT.

## Citing

```
@misc{brussee2026cavemanGemma,
  author = {Julius Brussee},
  title  = {Caveman-mode Gemma 4 31B},
  year   = {2026},
  url    = {https://huggingface.co/JBrussee/gemma-4-31B-caveman}
}
```

## See also

- Caveman skill (style source-of-truth): https://github.com/JuliusBrussee/caveman
- Agent / pitfall notes for this codebase: [`AGENTS.md`](AGENTS.md)
