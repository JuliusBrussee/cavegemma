#!/usr/bin/env bash
# Run this once on a fresh RunPod H100 80GB PCIe pod (Unsloth or PyTorch base image).
# After it completes, run: python training/train_unsloth.py --config training/config.toml

set -euo pipefail

echo ">>> System info"
nvidia-smi
python --version

echo ">>> Install Unsloth + deps"
# Unsloth handles its own torch / xformers / bitsandbytes pinning; let it.
pip install --break-system-packages --upgrade pip
pip install --break-system-packages "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --break-system-packages --no-deps "trl<0.18" peft accelerate bitsandbytes
pip install --break-system-packages datasets sentence-transformers wandb tomli rich tqdm

echo ">>> Auth"
# huggingface_hub auto-reads HF_TOKEN env var; no login required.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARN: HF_TOKEN not set — gated model downloads (Gemma 4) will 401."
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" || echo "wandb login failed (continuing without)"
else
    echo "WARN: WANDB_API_KEY not set. Set report_to='none' in config.toml or training disables metric logging."
fi

echo ">>> Sanity check"
python -c "import torch, unsloth, trl, peft, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'unsloth', unsloth.__version__, 'trl', trl.__version__, 'peft', peft.__version__)"

echo ">>> Done. Next:  python training/train_unsloth.py --config training/config.toml"
