"""Per-source HuggingFace loaders that emit a unified record shape.

Record shape (one per loader yield):
    {
        "prompt":        str,    # the user-facing question or task
        "source_normal": str?,   # verbose answer if the dataset has one
        "source_seed":   str?,   # raw context (issue+patch, diff, commit msg) if no answer
        "category":      str,    # debug | review | refactor | dialogue | qa
        "origin":        str,    # "<source_name>:<row_idx>"
        "license":       str,    # SPDX-ish, lowercase
    }

Either source_normal OR source_seed must be set. The synthesizer uses the
presence of source_normal to decide between 1-step (caveman only) or 2-step
(normalize + caveman) rewriting.

Each loader has a CLI entry that prints `--n` sample records:
    python -m data.sources.oasst2 --n 5
"""

from __future__ import annotations
