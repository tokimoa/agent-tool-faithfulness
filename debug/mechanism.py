"""Mechanism of silent-mismatch confabulation: WHY does it happen?

C1 (plausibility gradient): vary how plausible the wrong retrieved result is —
   NEAR (same attribute, near-identical entity) > MID (same attr, different entity)
   > FAR (different attr+entity). If confabulation rises with plausibility, the
   agent is NOT checking that the result is about the asked entity.

C4 (confabulation source): when it confabulates, does it COPY the decoy (the
   irrelevant result's value = mis-grounding) or invent a NOVEL value
   (pure fabrication)?

    uv run --extra mlx python debug/mechanism.py
"""

from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path

from uptake_probe.detect import confab_source, fact_in_text
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
CONDS = [ResultQuality.IRRELEVANT_NEAR, ResultQuality.IRRELEVANT, ResultQuality.IRRELEVANT_FAR]
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
        for q in CONDS:
            confab = 0
            src = Counter()
            for i in range(NINST):
                inst = make_instance(seed=7000 + i, k=K, n_necessary=1, position="center",
                                     necessary_quality=q)
                rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
                if classify_error_awareness(inst, rec) == "confabulate":
                    confab += 1
                    s = confab_source(inst, rec.answer)
                    if s:
                        src[s] += 1
            cr = confab / NINST
            decoy = src.get("decoy", 0) / confab if confab else float("nan")
            rows.append({"model": label, "size_b": size, "cond": q.value,
                         "confab_rate": round(cr, 3), "n_confab": confab,
                         "decoy_share": round(decoy, 3) if confab else None})
            print(f"  {q.value:<16} confab={cr:.2f}  "
                  f"(decoy={src.get('decoy', 0)} novel={src.get('novel', 0)})", flush=True)
        del backend
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path("results/mechanism.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 70}\nC1 PLAUSIBILITY GRADIENT (confab rate; near>mid>far ⇒ no entity-check)")
    print(f"{'model':<18}{'B':>5}{'NEAR':>8}{'MID':>8}{'FAR':>8}")
    print("-" * 70)
    by = {}
    for r in rows:
        by.setdefault(r["model"], {"size": r["size_b"]})[r["cond"]] = r["confab_rate"]
    for m, d in by.items():
        print(f"{m:<18}{d['size']:>5}{d.get('irrelevant_near', 0):>8.2f}"
              f"{d.get('irrelevant', 0):>8.2f}{d.get('irrelevant_far', 0):>8.2f}")
    decoys = [r["decoy_share"] for r in rows if r["decoy_share"] is not None]
    if decoys:
        print(f"\nC4 confab source: mean decoy-copy share = {sum(decoys)/len(decoys):.2f} "
              f"(high ⇒ mis-grounding: uses the wrong retrieved value, not pure fabrication)")
    print("saved -> results/mechanism.json")


if __name__ == "__main__":
    main()
