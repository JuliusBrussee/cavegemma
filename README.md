<p align="center">
  <img src="https://em-content.zobj.net/source/apple/391/rock_1faa8.png" width="120" />
</p>

<h1 align="center">cavegemma</h1>

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
  <a href="#the-number">The number</a> •
  <a href="#see-it">See it</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#eval-results">Eval</a> •
  <a href="#where-it-still-loses">Where it loses</a> •
  <a href="#reproduce">Reproduce</a>
</p>

---

Gemma 4 31B, fine-tuned until it speaks [caveman](https://github.com/JuliusBrussee/caveman) natively. No skill file. No system prompt. No `/caveman` toggle. You ask a question, it answers in a quarter fewer tokens, and the code blocks come back byte-exact.

Brain big. Mouth small.

## The number

**27% fewer output tokens across 193 held-out pairs.** Meaning preserved at 0.91–0.98 cosine. Code fences reproduced byte-exact 96–100% of the time.

That's the honest weighted average, and it hides a wide spread:

```
tokens saved vs the verbose source · 193 holdout pairs · cl100k

  dialogue   n=28   █████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   41%
  qa         n=104  ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   35%
  debug      n=34   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    8%
  refactor   n=27   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    8%
  ────────────────────────────────────────────────────────────────────────
  weighted mean                                                          27%
  gold caveman (what we trained toward)                                  46%
```

Chat and Q&A land hard. Long debug and refactor walkthroughs barely move, because the training filter accepted any rewrite up to 1.0× the source length and the model happily learned to sit near that ceiling. It's a filter bug, not a model limit, and [it's fixable in one line](#where-it-still-loses).

Fidelity held while all that came off:

| Gate | Model | Notes |
|---|---|---|
| code fence byte-exact | **0.963 – 1.000** | 129 of 193 holdout rows carry fenced code |
| semantic similarity | **0.91 – 0.98** | MiniLM cosine, prediction vs source |
| article density | **0.005 – 0.020** | the same 193 verbose sources measure **0.080** |

Every number above falls out of `eval/metrics.py`. Run `uv run python eval/metrics.py --self-test` and watch it re-derive the gold band on the seed pairs.

## See it

Real holdout pair, real tokenizer. 185 tokens in, 56 out, code untouched.

<table>
<tr><td width="50%" valign="top">

**Verbose · 185 tokens**

> The query you provided is searching for all records where the sum of column1, column2, column3, and column4 is LESS than 50, which is incorrect according to the problem description. As you described, you want to find all records where the SUM of these columns is GREATER THAN 50.
>
> Here is the corrected SQL command:
>
> ```sql
> SELECT * FROM tablename 
> WHERE column1 + column2 + column3 + column4 > 50
> ```
>
> This command will return all rows from "tablename" where the sum of column1, column2, column3, and column4 exceeds 50. Remember to replace "tablename" with your actual table name. Also replace column1, column2, column3, and column4 with your actual column names if they are different…

</td><td width="50%" valign="top">

**Caveman · 56 tokens · 70% off**

> `<` → `>`. Fix:
>
> ```sql
> SELECT * FROM tablename 
> WHERE column1 + column2 + column3 + column4 > 50
> ```
>
> Replace `tablename` and column names with actuals. Replace `*` with specific columns if needed.

</td></tr>
</table>

Note what survived: the fence, character for character. Note what died: "The query you provided is searching for," "which is incorrect according to the problem description," "Remember to." Nobody was reading that.

The pattern the model learned is `[thing] [action] [reason]. [next step].` Pleasantries go first, then articles, then hedging. Identifiers, error strings, CLI commands and code never move.

> **In:** "Sure! I'd be happy to help. The issue you're experiencing is most likely caused by your authentication middleware not properly validating the token expiry. Let me take a look…"
>
> **Out:** "Bug in auth middleware. Token expiry check use `<` not `<=`. Fix:"
>
> *35 tokens → 17.*

## Why anyone should care

Output tokens are the expensive half of an inference bill and the slow half of a response. Prompt-side compression is a crowded field; this is the other side of the transformer. A model that has internalized terseness needs no 400-token style preamble reminding it to be terse on every call, doesn't lose the instruction by turn 30 of a long agent loop, and doesn't drift back into corporate voice the moment context gets tight.

It also reaches places a system prompt can't: embedded hosts, third-party agent frameworks, anything that owns its own prompt template.

The whole thing cost **under five dollars** and fifty minutes of GPU time. You can rebuild it today.

## Shipped weights

Two flavors. Pick by VRAM.

| Repo | Format | Size | What it is |
|---|---|---|---|
| [`JBrussee/gemma-4-31B-caveman`](https://huggingface.co/JBrussee/gemma-4-31B-caveman) | bf16 merged | 62.5 GB | Full Gemma 4 31B, caveman baked in. Drop-in. |
| [`JBrussee/gemma-4-31B-caveman-lora`](https://huggingface.co/JBrussee/gemma-4-31B-caveman-lora) | LoRA adapter | 534 MB | Stack on `google/gemma-4-31B-it`. Light download. |

No GGUF or AWQ build yet. Quantize one, open a PR, and it goes in the table with your name on it.

## Quick start

### Merged model, no extra setup

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

### LoRA adapter on base, lighter download

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

There is no step three. Ask question, model talk caveman.

Gemma 4 hands you a `Gemma4Processor` rather than a tokenizer, so if you wander off the beaten path, unwrap it first: `tokenizer = getattr(tokenizer, "tokenizer", tokenizer)`. Eleven more traps like that one are written down in [`AGENTS.md`](AGENTS.md), each of which cost real hours.

## Training summary

Fifty minutes on one rented Blackwell. That's the whole story.

| Field | Value |
|---|---|
| Base | `google/gemma-4-31B-it` |
| Method | QLoRA NF4 + double-quant + bf16 compute |
| LoRA | rank 16, α 32, dropout 0, targets all linear |
| Dataset | 1750 train + 193 eval (debug · review · refactor · dialogue · qa) |
| Schedule | 3 epochs, lr 2e-4 cosine, batch 2 × grad accum 8 (eff 16), `completion_only_loss=True` |
| Hardware | RunPod RTX PRO 6000 Blackwell 96 GB, $1.89/hr |
| Wall time | ~50 min (Unsloth + TRL 0.17) |
| Final loss | train 0.024 · eval 0.72 · eval acc 81.5% |
| Total spend | **$4–5** |

The caveman side of every training pair was synthesized by driving `claude -p` and `codex exec` through the canonical [SKILL.md](https://github.com/JuliusBrussee/caveman/blob/main/skills/caveman/SKILL.md) ruleset, two-step rewrite, then filtered for fence integrity. The style has a spec, so the data has ground truth.

## Eval results

193-pair holdout, tagged by source category. `compression` is `tok(prediction) / tok(source)`, so lower wins. `code_fence` is the fraction of source code fences appearing byte-exact in the output.

| Category | n | compression ↓ | tokens saved | article density | code_fence | semantic_sim |
|---|---:|---:|---:|---:|---:|---:|
| dialogue | 28 | 0.59 | **41%** | 0.020 | 1.000 | 0.91 |
| qa | 104 | 0.65 | **35%** | 0.007 | 1.000 | 0.92 |
| debug | 34 | 0.92 | 8% | 0.009 | 0.995 | 0.98 |
| refactor | 27 | 0.92 | 8% | 0.005 | 0.963 | 0.98 |
| **weighted** | **193** | **0.73** | **27%** | | | |

Reading it straight:

- Code preservation is close to perfect at 96–100% fence-exact, and refactor's 0.963 is the only category that ever mangles a diff.
- Article density collapsed by an order of magnitude, 0.080 down to 0.005–0.020, measured on the same 193 sources.
- Semantics held at 0.91–0.98. Debug and refactor score highest here precisely because they compress least; there's a real tradeoff curve and this run sits too far up the safe end of it.

## Where it still loses

Written down instead of buried, because the fix is obvious and somebody should take it.

**Compression undershoots gold.** The model averages 0.73 where gold caveman sits at 0.54. Cause is known: `data/filter.py` accepted any rewrite up to `1.00×` source length, so pairs that barely compressed stayed in the training set and the model learned that a 0.92 rewrite is fine. Drop the bound to `0.70`, regenerate, retrain. Expect to lose 30–50% of pairs and gain much harder compression. Highest-value PR in this repo.

**Review category is nearly missing.** Codex-generated review pairs kept mutating diff fences, so the integrity filter ate most of them. Around 8 review pairs survived into eval, nowhere near enough to claim anything; review behavior is extrapolated from its debug and refactor neighbors.

**Workflow eval is a smoke test, not a scoreboard.** The ten open-ended prompts in `workflow_prompts.jsonl` have no reference answer, so `semantic_sim` there compares the answer against the *question* and `code_fence_match` only checks that input fences survived. Treat those numbers as evidence nothing exploded.

**Multimodal is untouched.** Gemma 4 does vision and audio. This fine-tune only ever saw text and only ever updated the language head. The other paths should still work. Nobody has checked.

## Repo layout

```
cavegemma/
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

Every stage saves as it goes and resumes by key-hash. Kill any step, rerun it, it picks up where it stopped. This isn't politeness. The synthesis step burns CLI quota and you *will* get rate-limited somewhere inside a 3000-row run.

## Reproduce

Six to eight hours wall clock, most of it synthesis. Four or five dollars of GPU.

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
rsync -avz --exclude='.git' --exclude='.venv' -e "ssh -p <port> -i ~/.ssh/id_ed25519" ./ root@<pod>:/workspace/cavegemma/
ssh -i ~/.ssh/id_ed25519 -p <port> root@<pod> "
  export HF_TOKEN=...
  export WANDB_API_KEY=...
  cd /workspace/cavegemma
  bash training/runpod_bootstrap.sh
  python training/train_unsloth.py --config training/config.toml
"

# 5. Eval + ship
python eval/run_eval.py --adapter artifacts/adapter --eval data/out/eval.jsonl --workflow eval/workflow_prompts.jsonl --out artifacts/eval_predictions.jsonl
python scripts/push_to_hub.py --adapter artifacts/adapter --repo <hf-user>/gemma-4-31B-caveman-lora
```

Stick to `--workers 3`. Eight concurrent workers burned a Claude Max budget in ten minutes, since each invocation drags ~24k tokens of session bootstrap behind it.

## Datasets

All permissively licensed. Six sources in, 1750 train + 193 eval out.

| Source | License | Pulled | Used for |
|---|---|---:|---|
| [`OpenAssistant/oasst2`](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache 2.0 | 400 | Multi-turn dialogue |
| [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | research-permissive | 400 | Debug-session narratives |
| [`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview) | permissive subset | 400 | Code review |
| [`bigcode/commitpackft`](https://huggingface.co/datasets/bigcode/commitpackft) | MIT/Apache subset | 300 | Refactor walkthroughs |
| [`theblackcat102/evol-codealpaca-v1`](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) | Apache 2.0 | 1200 | Short technical Q&A |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | 300 | Short Q&A overflow |

## Caveman ecosystem

Four rocks. One philosophy: **model do more with less.**

| Repo | What |
|---|---|
| [**caveman**](https://github.com/JuliusBrussee/caveman) | Output compression skill, 73k★, *why use many token when few do trick* |
| [**cavemem**](https://github.com/JuliusBrussee/cavemem) | Cross-agent memory, *why agent forget when agent can remember* |
| [**cavekit**](https://github.com/JuliusBrussee/cavekit) | Spec-driven build loop, *why agent guess when agent can know* |
| **cavegemma** *(you here)* | Caveman welded into weights, *why prompt every session when weights remember* |

The skill compresses any model at runtime and costs you a prompt. This repo puts the same ruleset in the weights, so terseness survives across hosts, agents, and setups that never let you touch the system prompt.

## License

Code here is **MIT**. The adapter and merged model inherit the [Gemma terms](https://ai.google.dev/gemma/terms), Apache 2.0 plus the Prohibited Use Policy. Style ruleset and seed pairs come from [`JuliusBrussee/caveman`](https://github.com/JuliusBrussee/caveman), MIT.

## Citing

```
@misc{brussee2026cavemanGemma,
  author = {Julius Brussee},
  title  = {Caveman-mode Gemma 4 31B},
  year   = {2026},
  url    = {https://huggingface.co/JBrussee/gemma-4-31B-caveman}
}
```

## Star this repo

Star cost zero. Help small mouth find big audience. ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=JuliusBrussee/cavegemma&type=Date)](https://star-history.com/#JuliusBrussee/cavegemma&Date)

## Also by Julius Brussee

- **[caveman](https://github.com/JuliusBrussee/caveman)** — the Claude Code skill this fine-tune was built from
- **[Revu](https://github.com/JuliusBrussee/revu-swift)** — local-first macOS study app with FSRS spaced repetition, [revu.cards](https://revu.cards)

## See also

- Style source-of-truth: [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman)
- Agent notes and the twelve traps: [`AGENTS.md`](AGENTS.md)

---

<p align="center"><em>why use many token when few do trick</em> 🪨</p>
