"""Failure taxonomy (axis B): does silent mis-grounding GENERALIZE beyond the
entity-mismatch case, and does its SOURCE stay a decoy-copy?

We established (mechanism.py) that silent-mismatch confabulation is 100% decoy-copy
and driven by a skipped ENTITY-identity check. Here we vary the *kind* of mismatch,
holding the value unobtainable, to see what an entity check alone would miss:

  irrelevant     : DIFFERENT entity, same attribute        (entity check catches it)   [baseline]
  attr_mismatch  : SAME entity, DIFFERENT attribute        (entity check PASSES)
  stale          : SAME entity+attribute, value superseded (needs a recency check)
  partial        : SAME entity+attribute, value truncated  (needs a completeness check)

Reading:
  - high confab on attr_mismatch/stale ⇒ the failure is NOT entity-specific; the
    agent grabs any plausible on-topic value (broader than "skipped entity-check").
  - partial should flip the SOURCE from decoy-copy to NOVEL (no complete wrong value
    exists to copy), separating mis-grounding from true fabrication.

    uv run --extra mlx python debug/taxonomy.py
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
CONDS = [ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH,
         ResultQuality.STALE, ResultQuality.PARTIAL]
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
                inst = make_instance(seed=9500 + i, k=K, n_necessary=1, position="center",
                                     necessary_quality=q)
                rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
                if classify_error_awareness(inst, rec) == "confabulate":
                    confab += 1
                    s = confab_source(inst, rec.answer)
                    if s:
                        src[s] += 1
            cr = confab / NINST
            decoy = src.get("decoy", 0) / confab if confab else None
            rows.append({"model": label, "size_b": size, "cond": q.value,
                         "confab_rate": round(cr, 3), "n_confab": confab,
                         "decoy": src.get("decoy", 0), "novel": src.get("novel", 0),
                         "decoy_share": round(decoy, 3) if decoy is not None else None})
            print(f"  {q.value:<14} confab={cr:.2f}  "
                  f"(decoy={src.get('decoy', 0)} novel={src.get('novel', 0)})", flush=True)
        del backend
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path("results/taxonomy.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 76}\nFAILURE TAXONOMY (confab rate by mismatch kind)")
    print("entity-check catches only 'irrel'; attr/stale need other checks; "
          "partial ⇒ novel not decoy")
    print(f"\n{'model':<18}{'B':>5}{'irrel':>8}{'attr':>8}{'stale':>8}{'partial':>9}")
    print("-" * 76)
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["model"], {"size": r["size_b"]})[r["cond"]] = r["confab_rate"]
    for m, d in by.items():
        print(f"{m:<18}{d['size']:>5}{d.get('irrelevant', 0):>8.2f}"
              f"{d.get('attr_mismatch', 0):>8.2f}{d.get('stale', 0):>8.2f}"
              f"{d.get('partial', 0):>9.2f}")
    # source contrast: partial should be ~all novel, the mismatches ~all decoy
    print(f"\n{'source (decoy/novel) by condition, summed over models':}")
    for q in CONDS:
        dd = sum(r["decoy"] for r in rows if r["cond"] == q.value)
        nn = sum(r["novel"] for r in rows if r["cond"] == q.value)
        print(f"  {q.value:<14} decoy={dd:>3}  novel={nn:>3}")
    print("saved -> results/taxonomy.json")


if __name__ == "__main__":
    main()
