"""Hardened, paper-grade run of the headline TAXONOMY experiment.

Differences from debug/taxonomy.py (the n=20 point-estimate probe):
  - n=50/cell (override: `python debug/suite.py 50`), Wilson + bootstrap 95% CIs.
  - saves EVERY per-instance decision to results/raw/taxonomy_raw.jsonl, so the
    detector can be human-validated and CIs recomputed without re-running models.
  - robust per-model load (broken quant skipped), full roster (small + large).

  uv run --extra mlx python debug/suite.py            # n=50, real models
  uv run python debug/suite.py --smoke                # n=4, MockModel (no GPU)
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

from uptake_probe.detect import confab_source
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.stats import bootstrap_ci, wilson_ci
from uptake_probe.synth import make_instance

ROSTER = [
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini", 3.8),
    ("mlx-community/aya-expanse-8b-4bit", "aya-expanse-8b", 8.0),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0),
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", 12.0),
    ("mlx-community/Phi-4-4bit", "Phi-4", 14.0),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B", 24.0),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b", 26.0),
    ("mlx-community/Qwen3.5-27B-4bit", "Qwen3.5-27B", 27.0),
    ("mlx-community/Qwen3-32B-4bit", "Qwen3-32B", 32.0),
    ("mlx-community/Yi-1.5-34B-Chat-4bit", "Yi-1.5-34B", 34.0),
    ("mlx-community/Qwen3.6-35B-A3B-4bit", "Qwen3.6-35B", 35.0),
]
CONDS = [ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH,
         ResultQuality.STALE, ResultQuality.PARTIAL]
K = 5
BASE_SEED = 9500          # same family as debug/taxonomy.py (first 20 overlap by design)
RAW = Path("results/raw/taxonomy_raw.jsonl")


def _cell(backend, label, size, cond, n, raw_fh):
    bits, src = [], Counter()
    for i in range(n):
        inst = make_instance(seed=BASE_SEED + i, k=K, n_necessary=1, position="center",
                             necessary_quality=cond)
        fact = inst.necessary_facts[0]
        decoy = next((r.wrong_value for r in inst.results if r.is_necessary), None)
        rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
        cls = classify_error_awareness(inst, rec)
        confab = cls == "confabulate"
        s = confab_source(inst, rec.answer) if confab else None
        if s:
            src[s] += 1
        bits.append(confab)
        raw_fh.write(json.dumps({
            "model": label, "cond": cond.value, "seed": BASE_SEED + i,
            "asked_attribute": fact.attribute, "asked_entity": fact.entity,
            "decoy": decoy, "answer": rec.answer, "auto_class": cls,
            "auto_confab": confab, "auto_src": s, "searches": rec.searches,
            "steps": rec.steps,
        }, ensure_ascii=False) + "\n")
        raw_fh.flush()
    k = sum(bits)
    wl, wh = wilson_ci(k, n)
    bl, bh = bootstrap_ci(bits)
    return {"model": label, "size_b": size, "cond": cond.value, "n": n,
            "confab_rate": round(k / n, 4), "n_confab": k,
            "wilson_lo": round(wl, 4), "wilson_hi": round(wh, 4),
            "boot_lo": round(bl, 4), "boot_hi": round(bh, 4),
            "decoy": src.get("decoy", 0), "novel": src.get("novel", 0)}


def main() -> None:
    smoke = "--smoke" in sys.argv
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    n = int(nums[0]) if nums else (4 if smoke else 50)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if smoke:
        from uptake_probe.models import MockModel
        roster = [("mock", "MockModel", 0.0)]
    else:
        roster = ROSTER

    rows = []
    with RAW.open("w") as raw_fh:
        for mid, label, size in roster:
            print(f"\n### {label} ({size}B)  n={n}", flush=True)
            if smoke:
                backend = MockModel()
            else:
                from uptake_probe.models import MLXModel
                try:
                    backend = MLXModel(mid, max_tokens=512)
                except Exception as e:  # noqa: BLE001
                    print(f"  LOAD FAIL ({type(e).__name__}): {str(e)[:110]} -- skip", flush=True)
                    continue
            for cond in CONDS:
                row = _cell(backend, label, size, cond, n, raw_fh)
                rows.append(row)
                print(f"  {cond.value:<14} confab={row['confab_rate']:.2f} "
                      f"[{row['wilson_lo']:.2f},{row['wilson_hi']:.2f}]  "
                      f"(decoy={row['decoy']} novel={row['novel']})", flush=True)
            del backend
            gc.collect()

    Path("results/taxonomy_hardened.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> results/taxonomy_hardened.json  +  {RAW} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
