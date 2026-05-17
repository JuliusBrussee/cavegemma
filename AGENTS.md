# AGENTS.md

Project context + traps for future agents (Claude Code, Codex, Cursor, whatever) editing this repo.

## What this project is

LoRA + merged-bf16 fine-tune of `google/gemma-4-31B-it` to speak "caveman mode" natively. Style is defined by [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) (MIT). Status: shipped to HuggingFace as `JBrussee/gemma-4-31B-caveman` (bf16) and `JBrussee/gemma-4-31B-caveman-lora` (adapter). Trained May 17 2026 in ~50 min on a RunPod RTX PRO 6000 Blackwell 96GB.

The user is **Julius Brussee** — GitHub `JuliusBrussee`, HF `JBrussee`. Note the two handles don't match.

## What "caveman" means in code

The trained model rewrites or answers in caveman style: drops articles (a/an/the), drops filler (just/really/basically/actually/simply), drops pleasantries and hedging, allows fragments, follows the pattern `[thing] [action] [reason]. [next step].`. Critical invariant: **code blocks, function names, error strings, CLI commands stay byte-exact**.

You speak caveman in conversation with Julius. You DO NOT speak caveman in code comments, commits, PR descriptions, or security warnings. See [`feedback-caveman-mode`](../.claude/projects/-Users-julb-Desktop-GitHub-finetune-caveman/memory/feedback_caveman_mode.md) if memory is available.

## Pipeline at a glance

```
data/sources/*.py  →  data/build_corpus.py  →  data/synthesize.py  →  data/filter.py  →  data/split.py
                                                                                            ↓
                                  training/train_unsloth.py (Unsloth + TRL SFT, QLoRA NF4)
                                                                                            ↓
                                                  eval/run_eval.py  →  eval/judge.py
                                                                                            ↓
                                                       scripts/push_to_hub.py
```

Every stage is **save-as-you-go + resume by key-hash**. Kill any step at any time and rerun — it picks up where it left off. This is mandatory because the synthesis step burns through CLI quotas (`claude -p`, `codex exec`) and you WILL hit a rate limit mid-3000-row run.

## Specific traps (each cost real hours / $)

These are real things that broke during the build of this model. They will probably re-bite if you regenerate or retrain.

1. **TRL 0.17 API renames.** `SFTTrainer(tokenizer=...)` → `SFTTrainer(processing_class=...)`. `SFTConfig(assistant_only_loss=True)` does not exist on 0.17; only `completion_only_loss=True` is portable. The training script already handles this — don't revert.

2. **Gemma 4 multimodal processor.** Unsloth's `FastLanguageModel.from_pretrained("google/gemma-4-...")` returns a `Gemma4Processor` (text+vision+audio), not a tokenizer. Unwrap with `tokenizer = getattr(tokenizer, "tokenizer", tokenizer)` before passing to TRL or before calling `apply_chat_template` with plain-string content. Already done in `train_unsloth.py`, `eval/run_eval.py`, `scripts/infer.py`.

3. **`UNSLOTH_RETURN_LOGITS=1`.** Unsloth 2024.11+ returns empty logits by default. TRL's `compute_loss` needs real logits. Must set env var **before** `import unsloth`. `os.environ.setdefault` is unsafe — use unconditional assignment. Already pinned in `training/train_unsloth.py`.

4. **`hf upload` is single-stream and gets throttled.** For folders > 10GB use `hf upload-large-folder --num-workers 8`. Same 62 GB upload finished in 5 min vs the naive tool's projected 6+ hours.

5. **`bigcode/commitpackft` ships a script-based loader** that `datasets` v4+ refuses to run. Bypass: load the per-language `data.jsonl` URLs directly with `load_dataset("json", data_files=[urls], streaming=True)`. Already done in `data/sources/commitpack.py`.

6. **`ronantakizawa/github-codereview` column names** are `reviewer_comment` + `diff_context` (not `comment` + `diff_hunk`). p25 of `quality_score` ≈ 0.36, so filters at 0.4+ kill 25%+ of rows.

7. **CLI flag parsing on diff content.** Passing patch text containing `---` to `claude -p ---xyz` makes the CLI treat `---xyz` as an unknown flag and exit 1 with empty stderr. Always pipe content via stdin (`subprocess.run(cmd, input=content, ...)`), never as a positional arg. Already fixed in `data/synthesize.py`.

8. **HF datasets streaming hangs on process exit.** Background connection threads keep Python alive for minutes after the main loop ends. `data/build_corpus.py` calls `os._exit(0)` at the end to force exit. Don't remove that.

9. **Concurrent `claude -p` / `codex exec` rate-limits.** Each invocation incurs ~24k tokens of "session bootstrap" load. 8 concurrent workers blew Julius's Pro/Max budget in 10 min. Default in `data/synthesize.py` is now `--workers 3`. Codex also has a sliding-window burst limit; bursts of 100+ calls trigger a 6-hour cooldown.

10. **RunPod / Ubuntu 24.04 PEP 668.** Pip refuses system-level installs by default. `training/runpod_bootstrap.sh` uses `--break-system-packages` since the pod is disposable. Don't replace that with a venv unless you want to also rewrite the bootstrap.

11. **`huggingface-cli login` is deprecated.** Use `HF_TOKEN` env var only. The old command exits non-zero on the deprecation path, killing any `set -e` bash wrapper.

12. **tmux sessions die silently** when the script under them exits non-zero with `set -e`. After kicking off a long bg job, always `sleep 3 && tmux ls` to verify it's still alive. Don't trust the launch echo.

## How to extend the dataset (without retraining)

If you want to add more pairs:

```bash
# Add or change a loader in data/sources/<name>.py implementing iter_records(limit)
# yielding {prompt, source_normal?, source_seed?, category, origin, license}.

# Register in data/build_corpus.py's DEFAULT_QUOTAS.

# Run only that source:
uv run python data/build_corpus.py --only <name> --limit 400
uv run python data/synthesize.py --backend claude --workers 3  # resumes
uv run python data/filter.py --in data/out/raw_pairs.jsonl --out data/out/clean_pairs.jsonl
uv run python data/split.py --in data/out/clean_pairs.jsonl
```

Then retrain with `training/train_unsloth.py --config training/config.toml`. `resume_from_checkpoint=True` is auto-detected via `out_dir.glob("checkpoint-*")`.

## How to tighten compression

Current model compresses ~10-40% rather than the gold 50-70%. Filter upper bound in `data/filter.py` is `1.00` — relax to e.g. `0.75`, regenerate filtered set, retrain. Expect to lose 30-50% of pairs but gain harder compression.

## How to add a new model variant (e.g. Gemma 3 4B)

Change `training/config.toml`'s `model.base` and `model.max_seq_length`. Confirm Unsloth supports the variant (check https://unsloth.ai/docs). Smaller models can use fewer epochs and higher learning rate (e.g. 5e-4). Keep `completion_only_loss=True`.

## Important external references

- Style ruleset: https://github.com/JuliusBrussee/caveman/blob/main/skills/caveman/SKILL.md
- Aligned baseline↔caveman seeds (10 pairs): https://raw.githubusercontent.com/JuliusBrussee/caveman/main/evals/snapshots/results.json
- Unsloth docs: https://unsloth.ai/docs
- TRL docs (mind version pinning): https://huggingface.co/docs/trl
- Gemma 4 docs: https://ai.google.dev/gemma

## Plan file from initial design

`/Users/julb/.claude/plans/jaunty-doodling-treehouse.md` holds the original plan and v2 update for posterity. Decisions captured there:
- Pick 31B Dense, skip the 26B-A4B MoE (Unsloth bnb-4bit doesn't support MoE fused 3D expert tensors)
- QLoRA NF4 + bf16 compute, rank 16, all linear targets
- Workflow-rich data over short Q&A (1500 workflow + 1500 qa, mixed)
- Per-category compression bands
- Code-fence integrity is a hard filter and an eval gate

## Don't

- Don't store API keys in this repo. They've been leaked once in chat history (the WANDB key Julius pasted, the HF token I echoed via `cat env.sh`). Both should be rotated; never write either to a tracked file.
- Don't merge in untracked sibling directories (`blackjack/`, `crm/`, `smtp-microservice/`) — they're unrelated noise from Julius's working dir.
- Don't commit anything under `data/out/`, `artifacts/`. Already gitignored.

## Want to ship a new model?

1. Bump version in pyproject.toml and HF repo name (e.g. `gemma-4-31B-caveman-v2`).
2. Retrain. Push to a NEW HF repo, don't overwrite v1 (Julius cares about reproducibility).
3. Update `README.md`'s eval table.
4. Tag a release in GitHub.
