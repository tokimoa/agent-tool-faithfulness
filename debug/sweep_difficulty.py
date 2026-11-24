"""Find the degradation region: sweep N (necessary facts) at fixed K.

Loads the model once and sweeps so we can see where conjunctive
Necessary-Evidence Coverage starts to fall off the ceiling (the §14 guardrail-2
operating point). Run:

    uv run --extra mlx python debug/sweep_difficulty.py [model_id] [N_csv] [K] [n_inst]
    uv run --extra mlx python debug/sweep_difficulty.py mlx-community/Phi-4-mini-instruct-4bit 3,5,8 12 15
"""

from __future__ import annotations

import sys

from uptake_probe.models import MLXModel
from uptake_probe.pilot import _print_table, run_pilot

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-9B-MLX-4bit"
N_VALUES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 3, 5, 8]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 12
N_INST = int(sys.argv[4]) if len(sys.argv) > 4 else 15


def main() -> None:
    backend = MLXModel(MODEL_ID, max_tokens=384)
    print(f"loaded: {backend.name}\n")
    for n in N_VALUES:
        rep = run_pilot(backend, ks=(K,), modes=("passive", "agentic"),
                        n_instances=N_INST, n_necessary=n)
        print(f"\n### N_necessary = {n}  (K={K}, center)")
        _print_table(rep)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
