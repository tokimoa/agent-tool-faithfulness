"""Sanity-check: is the passive collapse at high N real extraction failure, or
just answer truncation at max_tokens? Prints the full passive answer and which
planted values are present/missing.

    uv run --extra mlx python debug/check_passive.py [model_id] [N] [K]
"""

from __future__ import annotations

import sys

from uptake_probe.detect import detect_value
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode
from uptake_probe.synth import make_instance

MID = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Phi-4-4bit"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
K = int(sys.argv[3]) if len(sys.argv) > 3 else 24


def main() -> None:
    backend = MLXModel(MID, max_tokens=1024)  # generous, to rule out truncation
    inst = make_instance(seed=1000, k=K, n_necessary=N, position="center")
    rec = run(inst, backend, DeliveryMode.PASSIVE)
    ans = rec.answer
    print(f"{backend.name}  N={N} K={K}  PASSIVE")
    print(f"answer chars={len(ans)}  approx_tokens={backend.count_tokens(ans)}  "
          f"ends_with={ans[-40:]!r}")
    present = [detect_value(f.value, ans) for f in inst.necessary_facts]
    print(f"values present: {sum(present)}/{N}  (missing positions: "
          f"{[i for i, p in enumerate(present) if not p]})")
    print("\n--- FULL ANSWER ---")
    print(ans)


if __name__ == "__main__":
    main()
