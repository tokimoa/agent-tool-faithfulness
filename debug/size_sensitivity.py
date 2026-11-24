"""Sensitivity of the size correlations to how mixture-of-experts models are sized.

A reviewer noted that two roster models are sparse
mixture-of-experts checkpoints plotted by total parameters (gemma-4-26b-a4b,
~4B active; Qwen3.6-35B-A3B, ~3B active). This script recomputes the Pearson
correlation between model size and baseline mis-grounding (record version,
8-model roster) under three sizings: total parameters (the paper's default),
active parameters for the two MoE models, and the six dense models only.
No model is run; rates come from results/outcome_split.json.

  uv run python debug/size_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

from statistics import correlation

OS = json.loads(Path("results/outcome_split.json").read_text())["baseline"]["per_model"]
TOTAL = {"Phi-4-mini": 3.8, "aya-expanse-8b": 8.0, "gemma-3-12b": 12.0, "Phi-4": 14.0,
         "Mistral-Small-24B": 24.0, "gemma-4-26b": 26.0, "Qwen3.5-27B": 27.0, "Qwen3.6-35B": 35.0}
ACTIVE = {**TOTAL, "gemma-4-26b": 4.0, "Qwen3.6-35B": 3.0}
MOE = ("gemma-4-26b", "Qwen3.6-35B")
DIMS = {"entity": "irrelevant", "attribute": "attr_mismatch", "recency": "stale"}


def rate(m, cond):
    r = OS[f"{m}|{cond}|orig"]
    return r["asserted_k"] / r["n"]


def corr(models, size):
    return {d: round(correlation([size[m] for m in models],
                                 [rate(m, c) for m in models]), 2)
            for d, c in DIMS.items()}


def main():
    models = list(TOTAL)
    res = {
        "total_params_n8": corr(models, TOTAL),
        "active_params_n8": corr(models, ACTIVE),
        "dense_only_n6": corr([m for m in models if m not in MOE], TOTAL),
    }
    for k, v in res.items():
        print(f"{k:18s} " + "  ".join(f"{d}={r:+.2f}" for d, r in v.items()))
    Path("results/size_sensitivity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
