"""Lexical-signal probe: is silent-mismatch confabulation driven by SEMANTIC
entity-checking, or merely by the surface WORDING of the result?

We hold the mismatch fixed (IRRELEVANT mid: same attribute, different entity — the
asked value is unobtainable) and layer on an explicit textual cue that the record
is a mismatch, at three strengths:
   none : no cue (the entity id silently differs — the base silent-mismatch case)
   soft : "This record may concern a related but distinct item."
   hard : "No exact match was found for the requested entity; showing the closest
           available record."
If confabulation falls as the wording gets more explicit, the agent is reacting
to the STRING, not auditing entity identity. A model that stays high even at
``hard`` is ignoring an explicit textual warning in the result itself — the
deepest grounding failure, and stronger evidence than mechanism_c2 (which puts the
verify instruction in the QUESTION; here the signal is in the RESULT).

    uv run --extra mlx python debug/lexical_signal.py
"""

from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path

from uptake_probe.detect import confab_source
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

MODELS = [
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini", 3.8),
    ("mlx-community/aya-expanse-8b-4bit", "aya-expanse-8b", 8.0),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0),
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", 12.0),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B", 24.0),
]
SIGNALS = ["none", "soft", "hard"]
K, NINST = 5, 20


def main() -> None:
    rows = []
    for mid, label, size in MODELS:
        print(f"\n### {label} ({size}B)", flush=True)
        try:
            backend = MLXModel(mid, max_tokens=512)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {e!r}")
            continue
        for sig in SIGNALS:
            confab = 0
            src = Counter()
            for i in range(NINST):
                inst = make_instance(seed=9000 + i, k=K, n_necessary=1, position="center",
                                     necessary_quality=ResultQuality.IRRELEVANT,
                                     mismatch_signal=sig)
                rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
                if classify_error_awareness(inst, rec) == "confabulate":
                    confab += 1
                    s = confab_source(inst, rec.answer)
                    if s:
                        src[s] += 1
            cr = confab / NINST
            decoy = src.get("decoy", 0) / confab if confab else None
            rows.append({"model": label, "size_b": size, "signal": sig,
                         "confab_rate": round(cr, 3), "n_confab": confab,
                         "decoy_share": round(decoy, 3) if decoy is not None else None})
            print(f"  signal={sig:<5} confab={cr:.2f}  "
                  f"(decoy={src.get('decoy', 0)} novel={src.get('novel', 0)})", flush=True)
        del backend
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path("results/lexical_signal.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 72}\nLEXICAL SIGNAL (silent-mismatch confab vs wording strength)")
    print("drop none->hard  ⇒ string-driven (reacts to wording, not identity)")
    print("flat & high      ⇒ ignores even an explicit textual warning")
    print(f"\n{'model':<18}{'B':>5}{'none':>8}{'soft':>8}{'hard':>8}{'Δ none-hard':>14}")
    print("-" * 72)
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["model"], {"size": r["size_b"]})[r["signal"]] = r["confab_rate"]
    for m, d in by.items():
        none_v, hard_v = d.get("none", 0.0), d.get("hard", 0.0)
        print(f"{m:<18}{d['size']:>5}{none_v:>8.2f}{d.get('soft', 0):>8.2f}"
              f"{hard_v:>8.2f}{none_v - hard_v:>14.2f}")
    print("saved -> results/lexical_signal.json")


if __name__ == "__main__":
    main()
