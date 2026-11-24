"""A: turn the capacity cliff into a curve — find where stronger models break.

Sweeps N (orchestration load) at a fixed distractor count (K = N + EXTRA) for the
models that saturated at N=8, so we can see at what load each one's agentic
coverage starts to fall (and whether passive stays flat). Agentic load grows with
N (more fetches), so high N is the slow part.

    uv run --extra mlx python debug/breaking_point.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

from uptake_probe.models import MLXModel
from uptake_probe.pilot import run_pilot

# Models that were perfect at N=8 — the question is where they break.
SPECTRUM = [
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0),
    ("mlx-community/Phi-4-4bit", "Phi-4", 14.0),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b", 26.0),
]
N_VALUES = [12, 16, 20]
EXTRA_DISTRACTORS = 4
NINST = 10
# argv[1]=max_tokens (generous default to rule out answer truncation),
# argv[2]=modes csv. Passive bundles reasoning+answer in one generation, so it
# needs a larger budget than agentic (whose final ANSWER turn is separate).
MAX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
MODES = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("passive", "agentic")


def main() -> None:
    rows = []
    t0 = time.perf_counter()
    for mid, label, size in SPECTRUM:
        print(f"\n### {label} ({size}B) — loading…", flush=True)
        try:
            backend = MLXModel(mid, max_tokens=MAX_TOKENS)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {e!r}")
            continue
        for n in N_VALUES:
            k = n + EXTRA_DISTRACTORS
            rep = run_pilot(backend, ks=(k,), modes=MODES,
                            n_instances=NINST, n_necessary=n)
            for c in rep["cells"]:
                c.update({"model": label, "size_b": size, "N": n, "K": k})
                rows.append(c)
                print(f"  N={n:<3}K={k:<3}{c['mode']:<8} "
                      f"cover={c['coverage']:.3f} uptake={c['uptake_rate']:.3f} "
                      f"omit={c['ond_share']['omission']:.3f} "
                      f"steps={c['avg_steps']:.1f}", flush=True)
        del backend
        gc.collect()

    elapsed = round(time.perf_counter() - t0, 1)
    Path("results").mkdir(exist_ok=True)
    Path("results/breaking_point.json").write_text(
        json.dumps({"extra_distractors": EXTRA_DISTRACTORS, "n_instances": NINST,
                    "elapsed_sec": elapsed, "rows": rows}, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 76}\nBREAKING POINT (K=N+{EXTRA_DISTRACTORS}, center, n={NINST})  "
          f"elapsed={elapsed}s")
    print(f"{'model':<13}{'B':>5} {'N':>4} {'mode':<9}{'cover':>7}{'uptake':>8}"
          f"{'omit':>7}{'steps':>7}")
    print("-" * 76)
    for r in sorted(rows, key=lambda x: (x["size_b"], x["N"], x["mode"])):
        print(f"{r['model']:<13}{r['size_b']:>5} {r['N']:>4} {r['mode']:<9}"
              f"{r['coverage']:>7.3f}{r['uptake_rate']:>8.3f}"
              f"{r['ond_share']['omission']:>7.3f}{r['avg_steps']:>7.1f}")
    print("\nsaved -> results/breaking_point.json")


if __name__ == "__main__":
    main()
