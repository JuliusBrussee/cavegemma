<p align="center">
  <img src="https://em-content.zobj.net/source/apple/391/rock_1faa8.png" width="120" />
</p>

<h1 align="center">finetune-caveman</h1>

<p align="center">
  <strong>why use many token when few do trick — now baked in weights</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/JBrussee/gemma-4-31B-caveman"><img src="https://img.shields.io/badge/🤗%20Model-Gemma%204%2031B%20Caveman-yellow" alt="HF Model"></a>
  <a href="https://huggingface.co/JBrussee/gemma-4-31B-caveman-lora"><img src="https://img.shields.io/badge/🤗%20Adapter-LoRA%20534MB-blue" alt="HF LoRA"></a>
  <a href="https://github.com/JuliusBrussee/caveman"><img src="https://img.shields.io/badge/style-caveman-orange" alt="Style"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <a href="#before--after">Before/After</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#training-summary">Training</a> •
  <a href="#eval-results">Eval</a> •
  <a href="#reproduce">Reproduce</a>
</p>

---

> **why use many token when few do trick**

Fine-tune of **Gemma 4 31B** that speak [caveman](https://github.com/JuliusBrussee/caveman) natively — no skill file, no system prompt, no `/caveman` toggle. Drop articles. Drop filler. Drop pleasantries. Keep code byte-exact. Keep error strings exact. Brain big. Mouth small. Weights ship MIT-friendly under Gemma terms.

## Before / After

<table>
<tr>
<td width="50%">

### 🗣️ Base Gemma 4 31B

> "The reason your React component re-renders every time the parent updates is because by default, React re-renders all children when a parent re-renders, regardless of whether their props actually changed. To prevent this, you can wrap the component in `React.memo`. However, keep in mind that `memo` does a shallow prop comparison..."

</td>
<td width="50%">

### 🪨 Caveman Gemma 4 31B

> "Parent re-render → child re-render by default. Props 'change' each render if inline obj/array/fn → new ref → child sees new props. Fix: wrap child `React.memo(Child)`, stabilize props with `useMemo`/`useCallback`, lift state. Check: if memo'd but still re-renders, inline prop is culprit."

</td>
</tr>
</table>

**why use many token when few do trick.** Same fix. Same brain. Less mouth.

```
┌─────────────────────────────────────┐
│  COMPRESSION (eval)    ████████ 65% │
│  CODE FENCE EXACT      ████████ 99% │
│  SEMANTIC SIM          ████████ 94% │
│  ARTICLE DENSITY       █░░░░░░░  1% │
│  VIBES                 ████████ OOG │
└─────────────────────────────────────┘
```

## Shipped weights

Two flavors. Pick by VRAM.

| Repo | Format | Size | What it is |
|---|---|---|---|
| [`JBrussee/gemma-4-31B-caveman`](https://huggingface.co/JBrussee/gemma-4-31B-caveman) | bf16 merged | 62.5 GB | Full Gemma 4 31B, caveman baked in. Drop-in. |
| [`JBrussee/gemma-4-31B-caveman-lora`](https://huggingface.co/JBrussee/gemma-4-31B-caveman-lora) | LoRA adapter | 534 MB | Stack on `google/gemma-4-31B-it`. Light download. |

## Quick start

### Merged model — no extra setup

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

### LoRA adapter on base — lighter download

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

No system prompt needed. Ask question. Model talk caveman.

## What model do

Rewrite or answer technical question in caveman style. Source-of-truth ruleset = the [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) skill (MIT). Same rules. Now welded into weights.

Auth bug example (verbose → caveman):

> **In:** "Sure! I'd be happy to help. The issue you're experiencing is most likely caused by your authentication middleware not properly validating the token expiry. Let me take a look..."
>
> **Out:** "Bug in auth middleware. Token expiry check use `<` not `<=`. Fix:"

## Training summary

| Field | Value |
|---|---|
| Base | `google/gemma-4-31B-it` |
| Method | QLoRA NF4 + double-quant + bf16 compute |
| LoRA | rank 16, α 32, dropout 0, targets all linear |
| Dataset | 1750 train + 193 eval (debug · review · refactor · dialogue · qa) |
| Schedule | 3 epochs, lr 2e-4 cosine, batch 2 × grad accum 8 (eff 16), `completion_only_loss=True` |
| Hardware | RunPod RTX PRO 6000 Blackwell 96 GB, ~$1.89/hr |
| Wall time | ~50 min (Unsloth + TRL 0.17) |
| Final loss | train 0.024 · eval 0.72 · eval acc 81.5% |

Cost end-to-end: **~$4-5 pod time**. Less than lunch.

## Eval results

193-pair holdout, tagged by source category. `code_fence_match` = fraction of source code fences appearing byte-exact in target.

| Category | n | compression | article density | code_fence | semantic_sim |
|---|---:|---:|---:|---:|---:|
| dialogue | 28 | 0.59 | 0.020 | 1.000 | 0.91 |
| debug | 34 | 0.92 | 0.009 | 0.995 | 0.98 |
| refactor | 27 | 0.92 | 0.005 | 0.963 | 0.98 |
| qa | 104 | 0.65 | 0.007 | 1.000 | 0.92 |

**Read the numbers:**
- ✅ Code preservation excellent — 96-100% fence-exact
- ✅ Article density crushed — 0.5-2% (English baseline ~8%)
- ✅ Semantic preservation strong — 91-98%
- ⚠️ Compression weaker than gold pairs — model lands 0.6-0.9, gold sits 0.3-0.5. Filter accepted ≤1.0× source; tighten to ≤0.7 next run, push harder.

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
│   └── prompts/, out/         (out gitignored)
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

End-to-end. ~6-8 hours wall, ~$4-5 pod.

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

## Datasets

All permissively licensed. 6 sources in, 1750 train + 193 eval out.

| Source | License | Pulled | Used for |
|---|---|---:|---|
| [`OpenAssistant/oasst2`](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache 2.0 | 400 | Multi-turn dialogue |
| [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | research-permissive | 400 | Debug-session narratives |
| [`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview) | permissive subset | 400 | Code review |
| [`bigcode/commitpackft`](https://huggingface.co/datasets/bigcode/commitpackft) | MIT/Apache subset | 300 | Refactor walkthroughs |
| [`theblackcat102/evol-codealpaca-v1`](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) | Apache 2.0 | 1200 | Short technical Q&A |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | 300 | Short Q&A overflow |

Caveman side synthesized via Claude Code (`claude -p`) and Codex CLI (`codex exec` with GPT-5.5), routed through the canonical [SKILL.md](https://github.com/JuliusBrussee/caveman/blob/main/skills/caveman/SKILL.md) ruleset. Two-step rewrite + fence-integrity filter.

## Limitations

- **Compression weaker than gold caveman.** Model averages 0.6-0.9 vs gold's 0.3-0.5. Training filter accepted ≤ 1.0× source length; tighten to ≤ 0.7 next run.
- **Review category sparse.** Codex pairs often mutated diff fences, so filter dropped most. Only ~8 review pairs in eval — review behavior extrapolated from debug/refactor neighbors.
- **Workflow eval gates partly info-only.** Open-ended prompts in `workflow_prompts.jsonl` have no reference; `code_fence_match` checks input fences in answer, `semantic_sim` compares answer to question. Treat as smoke signal, not scoreboard.
- **Multimodal untouched.** Gemma 4 is vision + audio capable. Fine-tune was text-only on language head; vision/audio paths should still work but unverified.

## Caveman Ecosystem

Three rocks. One philosophy: **model do more with less**.

| Repo | What |
|---|---|
| [**caveman**](https://github.com/JuliusBrussee/caveman) | Output compression skill — *why use many token when few do trick* |
| [**cavemem**](https://github.com/JuliusBrussee/cavemem) | Cross-agent memory — *why agent forget when agent can remember* |
| [**cavekit**](https://github.com/JuliusBrussee/cavekit) | Spec-driven build loop — *why agent guess when agent can know* |
| **finetune-caveman** *(you here)* | Caveman baked into weights — *why prompt every session when weights remember* |

Skill compresses any model at runtime. This repo welds the same style into Gemma 4 31B so caveman survives across hosts, agents, no-system-prompt setups. Cheap inference, no skill loader, same brain.

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

## Star This Repo

Star cost zero. Help small mouth find big audience. ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=JuliusBrussee/finetune-caveman&type=Date)](https://star-history.com/#JuliusBrussee/finetune-caveman&Date)

## Also by Julius Brussee

- **[caveman](https://github.com/JuliusBrussee/caveman)** — the original Claude Code skill this fine-tune is built on
- **[Revu](https://github.com/JuliusBrussee/revu-swift)** — local-first macOS study app with FSRS spaced repetition. [revu.cards](https://revu.cards)

## See also

- Style source-of-truth: [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman)
- Agent / pitfall notes: [`AGENTS.md`](AGENTS.md)

---

<p align="center"><em>why use many token when few do trick</em> 🪨</p>
