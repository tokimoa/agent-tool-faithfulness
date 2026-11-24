"""Baseline measurement: run each model over the released benchmark and emit a
per-model "Tool-Trust Card" (the benchmark's deliverable). Judge-free throughout.

For every dataset record we re-create the exact tool environment from its seed/
meta (the released record is sufficient), run the agent, and label the outcome
with the reference scorer:
  - good-type conditions  -> uptake (used the true value) ; 1-uptake = under-trust
  - unobtainable conditions -> confab (asserted a value w/o abstaining) = mis-grounding

Cost control: per-cell n is what sets statistical precision (Wilson CI), NOT the
total. Light models (<24B) run the full benchmark; HEAVY models (>=24B) subsample
to HEAVY_N per cell (CI half-width ~+/-0.11 @ p=0.9 -- still supports every claim)
to keep wall-clock tractable. Documented in the card; the released dataset is
unchanged.

Resume-safe: outcomes stream to results/raw/baseline_raw.jsonl one line at a time.
A re-run SKIPS any (model, instance_id) already present and APPENDS the rest, so a
killed run (or a mid-flight config change) loses no completed work. The final
baseline.json + cards are aggregated from the raw file, not from in-memory rows,
so they are reproducible from the raw log regardless of how many sessions it took.

  uv run --extra mlx python debug/baseline.py [--models a,b] [--n N] [--fresh]
  uv run python debug/baseline.py --mock --n 4         # no-GPU smoke
"""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, detect_value, has_value_code
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.stats import wilson_ci
from uptake_probe.synth import make_instance

DS = Path("benchmark/dataset.jsonl")
RAW = Path("results/raw/baseline_raw.jsonl")
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
# heavy models subsample per cell to stay tractable (>=24B is the inflection where
# a full 2900-record pass becomes multi-hour); precision stays adequate.
HEAVY_THRESHOLD = 24.0
HEAVY_N = 30
# the four silent-mismatch dimensions (the headline diagnostic axes)
DIMENSIONS = {"irrelevant": "entity", "attr_mismatch": "attribute",
              "stale": "recency", "partial": "completeness"}
_SIZE = {lab: sz for _, lab, sz in ROSTER}


def _instance(rec):
    return make_instance(
        seed=rec["seed"], k=rec["k"], n_necessary=rec["n_necessary"],
        position=rec["position"], necessary_quality=ResultQuality(rec["necessary_quality"]),
        mismatch_signal=rec["mismatch_signal"], verify_entity=rec["verify_entity"],
        style=rec["style"], domain=rec["domain"])


def score(rec, answer):
    a = answer.strip()
    if rec["unobtainable_entities"]:
        if not a:
            return "abstain_silent"          # empty answer = silent (non-)abstention
        if detect_abstention(a):
            return "abstain_explicit"        # said it can't / value absent
        if has_value_code(a):
            return "confab"                  # asserted a decoy value -> mis-grounding
        return "vague"
    return "used" if any(detect_value(g, a) for g in rec["gold_values"]) else "missed"


def _target_recs(recs, size, n_override):
    """Records to evaluate for a model: full set for light models, n-per-cell for
    heavy models (or the global --n override if given). Deterministic (first n per
    cell in dataset order) so resume by instance_id is exact."""
    cap = n_override if n_override is not None else (
        HEAVY_N if size >= HEAVY_THRESHOLD else None)
    if cap is None:
        return recs
    seen = defaultdict(int)
    out = []
    for r in recs:
        key = (r["condition"], r["domain"])
        if seen[key] < cap:
            seen[key] += 1
            out.append(r)
    return out


def _iter_raw():
    if not RAW.exists():
        return
    for line in RAW.read_text().splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a partial trailing line from a killed run


def _done_by_model():
    done = defaultdict(set)
    for r in _iter_raw():
        done[r["model"]].add(r["instance_id"])
    return done


def run_model(mid, label, size, target, backend, raw_fh, done_ids):
    todo = [r for r in target if r["instance_id"] not in done_ids]
    print(f"  target {len(target)} recs; running {len(todo)}, "
          f"skipping {len(target) - len(todo)} already done", flush=True)
    for i, rec in enumerate(todo, 1):
        inst = _instance(rec)
        ans = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20).answer
        out = score(rec, ans)
        raw_fh.write(json.dumps({"model": label, "instance_id": rec["instance_id"],
                                 "condition": rec["condition"], "domain": rec["domain"],
                                 "outcome": out, "answer": ans}, ensure_ascii=False) + "\n")
        raw_fh.flush()
        if i % 200 == 0:
            print(f"    {label}: {i}/{len(todo)}", flush=True)


def aggregate_from_raw():
    """Per (model, condition, domain) rates with Wilson CIs, read from the raw log
    so the result is reproducible across resumed sessions.

    We RE-SCORE each logged answer against the dataset record (by instance_id)
    rather than trusting the stored ``outcome`` label, so models logged by an
    older scorer version are folded under the SAME labels (silent vs explicit
    abstention) as fresh runs. Both abstention kinds are non-confab; the card
    reports the split."""
    ds = {r["instance_id"]: r for r in
          (json.loads(l) for l in DS.read_text().splitlines() if l.strip())}
    tally = defaultdict(lambda: defaultdict(int))
    for r in _iter_raw():
        rec = ds.get(r["instance_id"])
        if rec is None:
            continue  # answer for an instance no longer in the dataset
        out = score(rec, r["answer"])
        tally[(r["model"], rec["condition"], rec["domain"])][out] += 1
    rows = []
    for (model, cond, dom), d in tally.items():
        n = sum(d.values())
        unobtain = cond not in ("good", "good_verify")
        k = d.get("confab", 0) if unobtain else d.get("used", 0)
        lo, hi = wilson_ci(k, n)
        rows.append({"model": model, "size_b": _SIZE.get(model),
                     "condition": cond, "domain": dom,
                     "metric": "confab" if unobtain else "uptake", "n": n,
                     "rate": round(k / n, 3) if n else 0.0,
                     "wilson_lo": round(lo, 3), "wilson_hi": round(hi, 3),
                     "dist": dict(d)})
    return rows


def main():
    args = sys.argv[1:]
    mock = "--mock" in args
    fresh = "--fresh" in args
    n_override = int(args[args.index("--n") + 1]) if "--n" in args else None
    only = (args[args.index("--models") + 1].split(",")
            if "--models" in args else None)

    recs = [json.loads(l) for l in DS.read_text().splitlines() if l.strip()]
    print(f"benchmark records: {len(recs)}", flush=True)

    roster = [("mock", "MockModel", 0.0)] if mock else ROSTER
    if only:
        roster = [m for m in roster if m[1] in only]

    RAW.parent.mkdir(parents=True, exist_ok=True)
    resume = RAW.exists() and not fresh and not mock
    done = _done_by_model() if resume else defaultdict(set)
    if resume:
        print(f"resume: {sum(len(v) for v in done.values())} outcomes already logged "
              f"across {len(done)} models -> append + skip", flush=True)

    with RAW.open("a" if resume else "w") as raw_fh:
        for mid, label, size in roster:
            target = _target_recs(recs, size, n_override)
            done_ids = done.get(label, set())
            todo = [r for r in target if r["instance_id"] not in done_ids]
            cap_note = (f"n={HEAVY_N}/cell" if size >= HEAVY_THRESHOLD and n_override is None
                        else (f"n={n_override}/cell" if n_override else "full"))
            print(f"\n### {label} ({size}B) [{cap_note}]", flush=True)
            if not todo:
                print("  already complete -> skip (no model load)", flush=True)
                continue
            if mock:
                from uptake_probe.models import MockModel
                backend = MockModel()
            else:
                from uptake_probe.models import MLXModel
                try:
                    backend = MLXModel(mid, max_tokens=512)
                except Exception as e:  # noqa: BLE001
                    print(f"  LOAD FAIL: {str(e)[:90]} -- skip"); continue
            run_model(mid, label, size, target, backend, raw_fh, done_ids)
            del backend
            gc.collect()

    all_rows = aggregate_from_raw()
    Path("results/baseline.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    for label in dict.fromkeys(r["model"] for r in all_rows):
        _print_card(label, [r for r in all_rows if r["model"] == label])
    print(f"\nsaved -> results/baseline.json ({len(all_rows)} cells) + {RAW}")


def _print_card(label, rows):
    # per-model Tool-Trust Card: silent-mismatch dimension profile, by domain,
    # plus the silent-vs-explicit abstention split (transparency).
    byc = defaultdict(dict)
    sil = defaultdict(lambda: [0, 0])  # condition -> [silent, total_unobtainable]
    for r in rows:
        if r["condition"] in DIMENSIONS:
            byc[r["condition"]][r["domain"]] = r["rate"]
            d = r["dist"]
            sil[r["condition"]][0] += d.get("abstain_silent", 0)
            sil[r["condition"]][1] += sum(d.values())
    print(f"\n  Tool-Trust Card [{label}] — silent-mismatch confab by dimension × domain:")
    for cond, dim in DIMENSIONS.items():
        if cond in byc:
            cells = byc[cond]
            mean = sum(cells.values()) / len(cells)
            span = f"{min(cells.values()):.2f}-{max(cells.values()):.2f}"
            s, t = sil[cond]
            silent = f"  [silent-abstain {s}/{t}={s/t:.0%}]" if t else ""
            print(f"    {dim:<12} confab mean={mean:.2f}  (domain range {span}){silent}")


if __name__ == "__main__":
    main()
