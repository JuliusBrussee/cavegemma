#!/usr/bin/env bash
# Run this once on a fresh RunPod H100 80GB PCIe pod (Unsloth or PyTorch base image).
# After it completes, run: python training/train_unsloth.py --config training/config.toml

set -euo pipefail

echo ">>> System info"
nvidia-smi
python --version

echo ">>> Install Unsloth + deps"
# Unsloth handles its own torch / xformers / bitsandbytes pinning; let it.
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.18" peft accelerate bitsandbytes
pip install datasets sentence-transformers wandb tomli rich tqdm

echo ">>> Auth"
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "WARN: HF_TOKEN not set. Run: huggingface-cli login"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "WARN: WANDB_API_KEY not set. Skipping wandb login (set report_to='none' in config.toml if you don't want it)."
fi

echo ">>> Sanity check"
python -c "import torch, unsloth, trl, peft, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'unsloth', unsloth.__version__, 'trl', trl.__version__, 'peft', peft.__version__)"

echo ">>> Done. Next:  python training/train_unsloth.py --config training/config.toml"
