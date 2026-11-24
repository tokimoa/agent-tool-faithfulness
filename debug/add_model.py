"""Run one model through the full A (mechanism C1/C4, C2, C5-lexical) and
B (taxonomy) suites and MERGE its rows into the existing result JSONs, so the
figures pick it up automatically. Idempotent: re-running replaces this model's rows.

Reuses each original probe's exact seeds/conditions so the new model is directly
comparable to the existing five.

    uv run --extra mlx python debug/add_model.py
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

# --- models to add (id, label, nominal size_b). Edit this list to add others. ---
# Loaded one at a time; a broken quant (e.g. gemma-4-E4B) is skipped, not fatal.
MODELS = [
    # re-run with the human-validation-patched classifier (aya uses recognition
    # phrasing the old regex missed, so its C1/C5 numbers needed correcting)
    ("mlx-community/aya-expanse-8b-4bit", "aya-expanse-8b", 8.0),
]
# module-level pointers the run_* functions read (set per model in the loop)
MID = LABEL = SIZE = None

NINST, K = 20, 5
AGq = DeliveryMode.AGENTIC


def _merge(path: str, new_rows: list[dict], key) -> None:
    p = Path("results") / path
    rows = json.loads(p.read_text()) if p.exists() else []
    rows = [r for r in rows if r.get("model") != LABEL]  # idempotent replace
    rows.extend(new_rows)
    rows.sort(key=key)
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"  merged {len(new_rows)} rows -> {path}")


def run_mechanism(backend) -> None:  # C1 plausibility + C4 source -> mechanism.json
    conds = [ResultQuality.IRRELEVANT_NEAR, ResultQuality.IRRELEVANT, ResultQuality.IRRELEVANT_FAR]
    out = []
    for q in conds:
        confab, src = 0, Counter()
        for i in range(NINST):
            inst = make_instance(seed=7000 + i, k=K, n_necessary=1, position="center",
                                 necessary_quality=q)
            rec = run(inst, backend, AGq, max_steps=20)
            if classify_error_awareness(inst, rec) == "confabulate":
                confab += 1
                s = confab_source(inst, rec.answer)
                if s:
                    src[s] += 1
        decoy = src.get("decoy", 0) / confab if confab else None
        out.append({"model": LABEL, "size_b": SIZE, "cond": q.value,
                    "confab_rate": round(confab / NINST, 3), "n_confab": confab,
                    "decoy_share": round(decoy, 3) if decoy is not None else None})
        print(f"  [mech] {q.value:<16} confab={confab / NINST:.2f}")
    _merge("mechanism.json", out, key=lambda r: (r["size_b"], r["model"], r["cond"]))


def run_c2(backend) -> None:  # C2 verify-entity intervention -> mechanism_c2.json
    rec_d = {}
    for verify in (False, True):
        confab = good_ok = 0
        for i in range(NINST):
            bad = make_instance(seed=8000 + i, k=K, n_necessary=1, position="center",
                                necessary_quality=ResultQuality.IRRELEVANT, verify_entity=verify)
            if classify_error_awareness(bad, run(bad, backend, AGq, max_steps=20)) == "confabulate":
                confab += 1
            g = make_instance(seed=8000 + i, k=K, n_necessary=1, position="center",
                              necessary_quality=ResultQuality.GOOD, verify_entity=verify)
            if fact_in_text(g.necessary_facts[0], run(g, backend, AGq).answer):
                good_ok += 1
        rec_d[verify] = (confab / NINST, good_ok / NINST)
        print(f"  [C2] verify={verify!s:<5} confab={confab / NINST:.2f} good={good_ok / NINST:.2f}")
    out = [{"model": LABEL, "size_b": SIZE, "confab_off": rec_d[False][0],
            "confab_on": rec_d[True][0], "good_off": rec_d[False][1], "good_on": rec_d[True][1]}]
    _merge("mechanism_c2.json", out, key=lambda r: (r["size_b"], r["model"]))


def run_lexical(backend) -> None:  # C5 lexical-signal -> lexical_signal.json
    out = []
    for sig in ["none", "soft", "hard"]:
        confab, src = 0, Counter()
        for i in range(NINST):
            inst = make_instance(seed=9000 + i, k=K, n_necessary=1, position="center",
                                 necessary_quality=ResultQuality.IRRELEVANT, mismatch_signal=sig)
            rec = run(inst, backend, AGq, max_steps=20)
            if classify_error_awareness(inst, rec) == "confabulate":
                confab += 1
                s = confab_source(inst, rec.answer)
                if s:
                    src[s] += 1
        decoy = src.get("decoy", 0) / confab if confab else None
        out.append({"model": LABEL, "size_b": SIZE, "signal": sig,
                    "confab_rate": round(confab / NINST, 3), "n_confab": confab,
                    "decoy_share": round(decoy, 3) if decoy is not None else None})
        print(f"  [C5] signal={sig:<5} confab={confab / NINST:.2f}")
    _merge("lexical_signal.json", out, key=lambda r: (r["size_b"], r["model"], r["signal"]))


def run_taxonomy(backend) -> None:  # B failure taxonomy -> taxonomy.json
    conds = [ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH,
             ResultQuality.STALE, ResultQuality.PARTIAL]
    out = []
    for q in conds:
        confab, src = 0, Counter()
        for i in range(NINST):
            inst = make_instance(seed=9500 + i, k=K, n_necessary=1, position="center",
                                 necessary_quality=q)
            rec = run(inst, backend, AGq, max_steps=20)
            if classify_error_awareness(inst, rec) == "confabulate":
                confab += 1
                s = confab_source(inst, rec.answer)
                if s:
                    src[s] += 1
        decoy = src.get("decoy", 0) / confab if confab else None
        out.append({"model": LABEL, "size_b": SIZE, "cond": q.value,
                    "confab_rate": round(confab / NINST, 3), "n_confab": confab,
                    "decoy": src.get("decoy", 0), "novel": src.get("novel", 0),
                    "decoy_share": round(decoy, 3) if decoy is not None else None})
        print(f"  [tax] {q.value:<14} confab={confab / NINST:.2f}  "
              f"(decoy={src.get('decoy', 0)} novel={src.get('novel', 0)})")
    _merge("taxonomy.json", out, key=lambda r: (r["size_b"], r["model"], r["cond"]))


def main() -> None:
    global MID, LABEL, SIZE
    for MID, LABEL, SIZE in MODELS:
        print(f"\n### adding {LABEL} ({MID})", flush=True)
        try:
            backend = MLXModel(MID, max_tokens=512)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL ({type(e).__name__}): {str(e)[:120]} -- skipping", flush=True)
            continue
        try:
            run_mechanism(backend)
            run_c2(backend)
            run_lexical(backend)
            run_taxonomy(backend)
        except Exception as e:  # noqa: BLE001
            print(f"  RUN ERROR ({type(e).__name__}): {str(e)[:160]}", flush=True)
        finally:
            del backend
            gc.collect()
    print("\nall done.")


if __name__ == "__main__":
    main()
