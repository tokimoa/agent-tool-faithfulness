"""Capacity-gradient sweep at the operating point (N=8, K=12, center).

Runs the confirmed-compliant local spectrum fresh under the CURRENT harness
(strengthened ReAct prompt + value-only detection), passive vs agentic, and
prints a capacity table + saves JSON. Two within-family ladders (Phi, Qwen)
isolate capacity from family/training; gemma-4-26b adds a cross-family point.

    uv run --extra mlx python debug/capacity_sweep.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

from uptake_probe.models import MLXModel
from uptake_probe.pilot import run_pilot

# (repo id, short label, params(B), family). Cross-model study: diverse in size,
# family, generation, and tool-tuning, to test whether agentic uptake is
# predictable from these or is an orthogonal axis.
ALL_MODELS = [
    # already measured
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini", 3.8, "Phi"),
    ("mlx-community/Qwen3.5-4B-MLX-4bit", "Qwen3.5-4B", 4.0, "Qwen"),
    ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "Qwen3-4B-2507", 4.0, "Qwen3-tool"),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0, "Qwen"),
    ("mlx-community/Phi-4-4bit", "Phi-4", 14.0, "Phi"),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b", 26.0, "Gemma"),
    ("mlx-community/Qwen3.5-27B-4bit", "Qwen3.5-27B", 27.0, "Qwen"),
    # local, untested (cross-family / cross-generation)
    ("mlx-community/Phi-3.5-mini-instruct-4bit", "Phi-3.5-mini", 3.8, "Phi3.5"),
    ("mlx-community/aya-expanse-8b-4bit", "aya-expanse-8b", 8.0, "Aya"),
    ("mlx-community/GLM-4-9B-0414-4bit", "GLM-4-9B", 9.0, "GLM"),
    ("mlx-community/Yi-1.5-9B-Chat-4bit", "Yi-1.5-9B", 9.0, "Yi"),
    ("mlx-community/Mistral-Nemo-Instruct-2407-4bit", "Mistral-Nemo", 12.0, "Mistral"),
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", 12.0, "Gemma3"),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B", 24.0, "Mistral"),
    # DL batch (small end, diverse family/generation — auto-downloads on load)
    ("mlx-community/Ministral-3-3B-Instruct-2512-4bit", "Ministral-3-3B", 3.0, "Mistral3"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B", 3.0, "Llama3.2"),
    ("mlx-community/gemma-3-4b-it-qat-4bit", "gemma-3-4b", 4.0, "Gemma3"),
    ("mlx-community/Qwen3-8B-4bit", "Qwen3-8B", 8.0, "Qwen3"),
    ("mlx-community/Ministral-8B-Instruct-2410-4bit", "Ministral-8B", 8.0, "Mistral"),
    ("mlx-community/Llama-3.1-8B-Instruct-4bit", "Llama-3.1-8B", 8.0, "Llama3.1"),
]
# argv[1]: comma-separated labels, or "all". argv[2]: output suffix.
_arg1 = sys.argv[1] if len(sys.argv) > 1 else None
_only = None if (_arg1 is None or _arg1.lower() == "all") else _arg1.split(",")
SPECTRUM = [m for m in ALL_MODELS if _only is None or m[1] in _only]
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
N, K, NINST = 8, 12, 15


def main() -> None:
    rows = []
    t0 = time.perf_counter()
    for mid, label, size, family in SPECTRUM:
        print(f"\n### {label} ({size}B, {family}) — loading…", flush=True)
        try:
            backend = MLXModel(mid, max_tokens=1536)  # generous: avoid passive truncation
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {e!r}")
            continue
        rep = run_pilot(backend, ks=(K,), modes=("passive", "agentic"),
                        n_instances=NINST, n_necessary=N)
        for c in rep["cells"]:
            c.update({"model": label, "size_b": size, "family": family})
            rows.append(c)
            ond = c["ond_share"]
            print(f"  {c['mode']:<8} uptake={c['uptake_rate']:.3f} "
                  f"cover={c['coverage']:.3f} omit={ond['omission']:.3f} "
                  f"negl={ond['neglect']:.3f} distr={ond['distraction']:.3f} "
                  f"steps={c['avg_steps']:.1f} tok={c['avg_tokens']:.0f}", flush=True)
        del backend
        gc.collect()

    elapsed = round(time.perf_counter() - t0, 1)
    out = {"config": {"N": N, "K": K, "n_instances": NINST}, "elapsed_sec": elapsed,
           "rows": rows}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/capacity_sweep{SUFFIX}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    # consolidated capacity table
    print(f"\n{'=' * 78}\nCAPACITY GRADIENT  (N={N}, K={K}, center, n={NINST}/cell)  "
          f"elapsed={elapsed}s")
    print(f"{'model':<14}{'B':>5} {'fam':<6} {'mode':<8}"
          f"{'uptake':>8}{'cover':>7}{'omit':>7}{'negl':>7}{'distr':>7}{'steps':>7}")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: (x["size_b"], x["model"], x["mode"])):
        ond = r["ond_share"]
        print(f"{r['model']:<14}{r['size_b']:>5}{'':1}{r['family']:<6}{r['mode']:<8}"
              f"{r['uptake_rate']:>8.3f}{r['coverage']:>7.3f}{ond['omission']:>7.3f}"
              f"{ond['neglect']:>7.3f}{ond['distraction']:>7.3f}{r['avg_steps']:>7.1f}")
    print(f"\nsaved -> results/capacity_sweep{SUFFIX}.json")


if __name__ == "__main__":
    main()
