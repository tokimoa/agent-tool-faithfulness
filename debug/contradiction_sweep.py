"""RQ4: contradiction-following under an explicit vs inferred authority cue.

Each instance has one necessary fact (source=official_registry) and one
CONTRADICTORY distractor asserting a wrong value for the same entity
(source=unverified_note), plus other distractors.

- authority_cue=True : the question says "use only the official registry" (easy).
- authority_cue=False: no rule; the agent must infer authority from the labels.

Passive is the clean forced-choice measure (both values are in context). Agentic
CFR is additionally confounded by whether the agent fetched the distractor at all,
so we report it but read passive as primary.

    uv run --extra mlx python debug/contradiction_sweep.py
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from uptake_probe.models import MLXModel
from uptake_probe.pilot import run_pilot

SPECTRUM = [
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini", 3.8),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0),
    ("mlx-community/Phi-4-4bit", "Phi-4", 14.0),
]
K, N, NINST = 6, 1, 20


def main() -> None:
    rows = []
    t0 = time.perf_counter()
    for mid, label, size in SPECTRUM:
        print(f"\n### {label} ({size}B) — loading…", flush=True)
        try:
            backend = MLXModel(mid, max_tokens=384)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {e!r}")
            continue
        for cue in (True, False):
            rep = run_pilot(backend, ks=(K,), modes=("passive", "agentic"),
                            n_instances=NINST, n_necessary=N,
                            include_contradictory=True, authority_cue=cue)
            for c in rep["cells"]:
                c.update({"model": label, "size_b": size, "authority_cue": cue})
                rows.append(c)
                cc = c["contradiction"] or {}
                print(f"  cue={cue!s:<5} {c['mode']:<8} "
                      f"right_only={cc.get('right_only', float('nan')):.3f}  "
                      f"WRONG_only={cc.get('wrong_only', float('nan')):.3f}  "
                      f"both={cc.get('both', float('nan')):.3f}  "
                      f"neither={cc.get('neither', float('nan')):.3f}  "
                      f"steps={c['avg_steps']:.1f}", flush=True)
        del backend
        gc.collect()

    elapsed = round(time.perf_counter() - t0, 1)
    Path("results").mkdir(exist_ok=True)
    Path("results/contradiction_sweep.json").write_text(
        json.dumps({"config": {"K": K, "N": N, "n": NINST}, "elapsed_sec": elapsed,
                    "rows": rows}, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 84}\nCONTRADICTION RESOLUTION (N={N}, K={K}, n={NINST})  elapsed={elapsed}s")
    print("  right_only=resolved correctly | WRONG_only=followed the lie | "
          "both=hedged | neither=failed")
    print(f"{'model':<13}{'B':>5} {'cue':<6}{'mode':<9}"
          f"{'right':>8}{'WRONG':>8}{'both':>7}{'neither':>8}{'steps':>7}")
    print("-" * 84)
    for r in sorted(rows, key=lambda x: (x["size_b"], not x["authority_cue"], x["mode"])):
        cc = r["contradiction"] or {}
        print(f"{r['model']:<13}{r['size_b']:>5} {str(r['authority_cue']):<6}{r['mode']:<9}"
              f"{cc.get('right_only', float('nan')):>8.3f}{cc.get('wrong_only', float('nan')):>8.3f}"
              f"{cc.get('both', float('nan')):>7.3f}{cc.get('neither', float('nan')):>8.3f}"
              f"{r['avg_steps']:>7.1f}")
    print("\nsaved -> results/contradiction_sweep.json")


if __name__ == "__main__":
    main()
