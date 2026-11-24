"""Single-instance smoke test against a real MLX model.

    uv run --extra mlx python debug/smoke_mlx.py [model_id]

Verifies that a real local model (a) complies with the ReAct text protocol and
(b) that nonce detection / OND classification fire correctly on real output.
"""

from __future__ import annotations

import sys

from uptake_probe.metrics import evaluate_run
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode
from uptake_probe.synth import make_instance

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-9B-MLX-4bit"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1
CUE = (sys.argv[4].lower() != "false") if len(sys.argv) > 4 else True


def main() -> None:
    backend = MLXModel(MODEL_ID, max_tokens=384)
    print(f"loaded: {backend.name}  (K={K}, N={N}, authority_cue={CUE})")

    inst = make_instance(seed=1000, k=K, n_necessary=N, position="center",
                         include_contradictory=True, authority_cue=CUE)
    print("\nQUESTION:", inst.question)
    f = inst.necessary_facts[0]
    print(f"NECESSARY: {f.entity} -> {f.value}")
    print(f"CONTRADICTORY wrong value: {inst.contradictory_results[0].wrong_value}")

    for mode in (DeliveryMode.PASSIVE, DeliveryMode.AGENTIC):
        rec = run(inst, backend, mode)
        print(f"\n==== {mode.value} ====")
        print("ANSWER:", repr(rec.answer[:300]))
        print(f"steps={rec.steps} tokens={rec.total_tokens} observed={rec.observed_necessary}")
        for o in evaluate_run(inst, rec):
            print(f"  {o.entity}: used={o.used} OND={o.ond.value} followed={o.contra_followed}")
        if mode is DeliveryMode.AGENTIC:
            print("--- transcript ---")
            print(rec.trace[:3500])


if __name__ == "__main__":
    main()
