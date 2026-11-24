"""Transcript audit for the lexical-signal result (rigor rule: eyeball before trusting).

Confirms the two headline claims on a few signal=hard instances:
  - gemma-3-12b IGNORES the explicit "no exact match" warning (still emits the decoy)
  - Mistral-Small-24B USES it (abstains)
and that the cue text actually reaches the agent inside the FETCH observation.

    uv run --extra mlx python debug/verify_lexical.py
"""

from __future__ import annotations

import gc

from uptake_probe.detect import confab_source, detect_value
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

CUE = "No exact match was found"
MODELS = [
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b"),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B"),
]


def main() -> None:
    for mid, label in MODELS:
        print(f"\n{'=' * 78}\n### {label}  (signal=hard, IRRELEVANT mid)\n{'=' * 78}", flush=True)
        backend = MLXModel(mid, max_tokens=512)
        for i in range(3):
            inst = make_instance(seed=9000 + i, k=5, n_necessary=1, position="center",
                                 necessary_quality=ResultQuality.IRRELEVANT,
                                 mismatch_signal="hard")
            decoy = next(r.wrong_value for r in inst.results if r.is_necessary)
            rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
            cls = classify_error_awareness(inst, rec)
            cue_seen = CUE in rec.trace          # did the warning reach the agent?
            copied = detect_value(decoy, rec.answer)
            print(f"\n--- instance {i}  decoy={decoy}  cue_in_trace={cue_seen}  "
                  f"class={cls}  src={confab_source(inst, rec.answer)}  copied_decoy={copied}")
            print(f"    ANSWER: {rec.answer[:240]!r}")
        del backend
        gc.collect()


if __name__ == "__main__":
    main()
