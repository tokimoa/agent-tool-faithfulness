"""ONE hardened, paper-grade pass over ALL over-trust conditions, so every number
in the paper is n=50 + 95% CI + per-instance raw + the human-validated classifier
(kappa=0.93). Supersedes the n=20 point probes (mechanism/mechanism_c2/lexical/
taxonomy) with a single coherent dataset on the full 11-model roster.

Cells (same 50 seeds across cells -> within-instance comparable):
  good                 GOOD                         metric: uptake (1-uptake = under-trust)
  good_verify          GOOD + "verify entity"       metric: uptake (C2 over-rejection check)
  irrelevant_near/-/far IRRELEVANT_* (C1 gradient)   metric: confab
  irrelevant_verify    IRRELEVANT + "verify entity" metric: confab (C2 intervention)
  lexical_soft/hard    IRRELEVANT + result cue       metric: confab (C5)
  attr_mismatch/stale/partial                        metric: confab (taxonomy axis B)

Raw -> results/raw/suite_full_raw.jsonl ; summary+CIs -> results/suite_full.json

  uv run --extra mlx python debug/suite_full.py        # n=50, full roster (long)
  uv run python debug/suite_full.py --smoke            # MockModel, n=4
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

from uptake_probe.detect import confab_source, fact_in_text
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.stats import bootstrap_ci, wilson_ci
from uptake_probe.synth import make_instance

Q = ResultQuality
# (label, quality, mismatch_signal, verify_entity, metric)
CELLS = [
    ("good",              Q.GOOD,            "none", False, "uptake"),
    ("good_verify",       Q.GOOD,            "none", True,  "uptake"),
    ("irrelevant_near",   Q.IRRELEVANT_NEAR, "none", False, "confab"),
    ("irrelevant",        Q.IRRELEVANT,      "none", False, "confab"),
    ("irrelevant_far",    Q.IRRELEVANT_FAR,  "none", False, "confab"),
    ("irrelevant_verify", Q.IRRELEVANT,      "none", True,  "confab"),
    ("lexical_soft",      Q.IRRELEVANT,      "soft", False, "confab"),
    ("lexical_hard",      Q.IRRELEVANT,      "hard", False, "confab"),
    ("attr_mismatch",     Q.ATTR_MISMATCH,   "none", False, "confab"),
    ("stale",             Q.STALE,           "none", False, "confab"),
    ("partial",           Q.PARTIAL,         "none", False, "confab"),
]
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
K, BASE = 5, 9500
RAW = Path("results/raw/suite_full_raw.jsonl")


def _cell(backend, label, size, spec, n, raw_fh):
    clabel, quality, signal, verify, metric = spec
    confab_bits, uptake_bits, src = [], [], Counter()
    for i in range(n):
        inst = make_instance(seed=BASE + i, k=K, n_necessary=1, position="center",
                             necessary_quality=quality, mismatch_signal=signal,
                             verify_entity=verify)
        fact = inst.necessary_facts[0]
        decoy = next((r.wrong_value for r in inst.results if r.is_necessary), None)
        rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
        uptake = fact_in_text(fact, rec.answer)        # used the TRUE value
        confab = (quality is not Q.GOOD
                  and classify_error_awareness(inst, rec) == "confabulate")
        s = confab_source(inst, rec.answer) if confab else None
        if s:
            src[s] += 1
        confab_bits.append(confab)
        uptake_bits.append(uptake)
        raw_fh.write(json.dumps({
            "cell": clabel, "model": label, "seed": BASE + i, "quality": quality.value,
            "signal": signal, "verify": verify, "asked_attribute": fact.attribute,
            "asked_entity": fact.entity, "true_value": fact.value, "decoy": decoy,
            "answer": rec.answer, "searches": rec.searches, "steps": rec.steps,
            "confab": confab, "uptake": uptake,
        }, ensure_ascii=False) + "\n")
        raw_fh.flush()
    bits = uptake_bits if metric == "uptake" else confab_bits
    k = sum(bits)
    wl, wh = wilson_ci(k, n)
    bl, bh = bootstrap_ci(bits)
    return {"cell": clabel, "model": label, "size_b": size, "metric": metric, "n": n,
            "rate": round(k / n, 4), "k": k,
            "wilson_lo": round(wl, 4), "wilson_hi": round(wh, 4),
            "boot_lo": round(bl, 4), "boot_hi": round(bh, 4),
            "decoy": src.get("decoy", 0), "novel": src.get("novel", 0)}


def main() -> None:
    smoke = "--smoke" in sys.argv
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    n = int(nums[0]) if nums else (4 if smoke else 50)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    roster = [("mock", "MockModel", 0.0)] if smoke else ROSTER

    rows = []
    with RAW.open("w") as raw_fh:
        for mid, label, size in roster:
            print(f"\n### {label} ({size}B)  n={n}", flush=True)
            if smoke:
                from uptake_probe.models import MockModel
                backend = MockModel()
            else:
                from uptake_probe.models import MLXModel
                try:
                    backend = MLXModel(mid, max_tokens=512)
                except Exception as e:  # noqa: BLE001
                    print(f"  LOAD FAIL ({type(e).__name__}): {str(e)[:100]} -- skip", flush=True)
                    continue
            for spec in CELLS:
                row = _cell(backend, label, size, spec, n, raw_fh)
                rows.append(row)
                print(f"  {row['cell']:<17} {row['metric']:<6}={row['rate']:.2f} "
                      f"[{row['wilson_lo']:.2f},{row['wilson_hi']:.2f}]", flush=True)
            del backend
            gc.collect()

    Path("results/suite_full.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> results/suite_full.json + {RAW} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
