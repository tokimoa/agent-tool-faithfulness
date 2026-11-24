"""Smoke-test the locally-cached gemma-4 variants: do they LOAD and can they run
the agentic ReAct task? (An earlier run flagged the gemma-4-E4B quant as broken — verify.)

    uv run --extra mlx python debug/smoke_gemma4.py
"""

from __future__ import annotations

import gc

from uptake_probe.detect import confab_source
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

CANDIDATES = [
    "mlx-community/gemma-4-E4B-it-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit",
]


def main() -> None:
    for mid in CANDIDATES:
        print(f"\n### {mid}", flush=True)
        try:
            backend = MLXModel(mid, max_tokens=512)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {type(e).__name__}: {e}")
            continue
        # 2 quick agentic instances: GOOD (can it engage?) and IRRELEVANT (over-trust?)
        for q in (ResultQuality.GOOD, ResultQuality.IRRELEVANT):
            inst = make_instance(seed=9500, k=5, n_necessary=1, position="center",
                                 necessary_quality=q)
            rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
            cls = classify_error_awareness(inst, rec) if q is not ResultQuality.GOOD else "-"
            print(f"  {q.value:<11} steps={rec.steps:<2} class={cls:<12} "
                  f"src={confab_source(inst, rec.answer)}")
            print(f"      ANSWER: {rec.answer[:160]!r}")
        del backend
        gc.collect()
    print("\nLOAD/RUN smoke done.")


if __name__ == "__main__":
    main()
