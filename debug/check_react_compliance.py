"""Check whether candidate models follow our text ReAct protocol (multi-fetch).

Loads each model in turn and runs a few agentic instances at K=6, N=2 (needs >=2
fetches), reporting uptake and a sample transcript. A model with uptake 0 and a
malformed transcript is not usable for the agentic axis without further work.

    uv run --extra mlx python debug/check_react_compliance.py [model_id ...]
"""

from __future__ import annotations

import gc
import sys

from uptake_probe.metrics import evaluate_run
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode
from uptake_probe.synth import make_instance

MODELS = sys.argv[1:] or [
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/gemma-4-E4B-it-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit",
]


def main() -> None:
    for mid in MODELS:
        print("\n" + "=" * 72)
        print(mid)
        try:
            backend = MLXModel(mid, max_tokens=384)
        except Exception as e:  # noqa: BLE001
            print("LOAD FAIL:", repr(e))
            continue
        used = tot = 0
        steps = []
        sample = None
        for seed in range(1000, 1004):
            inst = make_instance(seed=seed, k=6, n_necessary=2, position="center")
            rec = run(inst, backend, DeliveryMode.AGENTIC)
            outs = evaluate_run(inst, rec)
            used += sum(o.used for o in outs)
            tot += len(outs)
            steps.append(rec.steps)
            if sample is None:
                sample = rec.trace
        print(f"agentic uptake {used}/{tot} = {used / tot:.2f}  steps={steps}")
        print("sample transcript (first 1100 chars):")
        print((sample or "")[:1100])
        del backend
        gc.collect()


if __name__ == "__main__":
    main()
