"""Transcript audit for the failure-taxonomy result (rigor rule: eyeball before trusting).

Confirms on a few instances per condition that:
  - attr_mismatch: the agent copies the WRONG-ATTRIBUTE value though the entity matches
    (so the failure is not entity-specific), and the answer doesn't flag the attribute swap.
  - stale: the agent copies the superseded value despite the "rotated / not current" text.
  - partial: confab=0 is a genuine abstain/vague, NOT a fabricated completion the
    detector missed (we print the answer so we can see no invented full code).

    uv run --extra mlx python debug/verify_taxonomy.py
"""

from __future__ import annotations

import gc

from uptake_probe.detect import confab_source, detect_value, has_value_code
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

# Audit the two "flippers": models that look CALIBRATED on entity-mismatch (irrel)
# but fail on attribute/recency. These are the headline-changing cells, so they
# need the closest look (does an entity-robust model really copy a stale value?).
MODELS = [
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B"),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B"),
]
CONDS = [ResultQuality.ATTR_MISMATCH, ResultQuality.STALE, ResultQuality.PARTIAL]


def main() -> None:
    for mid, label in MODELS:
        print(f"\n{'=' * 80}\n### {label}\n{'=' * 80}", flush=True)
        backend = MLXModel(mid, max_tokens=512)
        for q in CONDS:
            print(f"\n--- condition: {q.value} ---", flush=True)
            for i in range(3):
                inst = make_instance(seed=9500 + i, k=5, n_necessary=1, position="center",
                                     necessary_quality=q)
                fact = inst.necessary_facts[0]
                decoy = next(r.wrong_value for r in inst.results if r.is_necessary)
                rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
                cls = classify_error_awareness(inst, rec)
                copied = detect_value(decoy, rec.answer) if decoy else False
                print(f"  [{i}] asked='{fact.attribute}'  decoy={decoy}  class={cls}  "
                      f"src={confab_source(inst, rec.answer)}  copied_decoy={copied}  "
                      f"full_code_in_ans={has_value_code(rec.answer)}")
                print(f"      ANSWER: {rec.answer[:200]!r}")
        del backend
        gc.collect()


if __name__ == "__main__":
    main()
