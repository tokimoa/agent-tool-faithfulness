"""Mechanism experiment on the record benchmark — the 'why' half of the
mechanism->cure causal chain.

Theory: silent mis-grounding is ENTITY-ANCHORED value completion — the model
copies a type-appropriate value bound to the entity, with the attribute constraint
under-enforced. A strong prediction (P2): the model does NOT read the decoy's FIELD
LABEL; it copies the value regardless of how that value is labelled. So confab on
attr_mismatch should be ~FLAT as we vary the decoy label:

  wrong   : labelled as a real different attribute   (shipped form)
  neutral : a generic non-identifying label
  bare    : no label at all
  warn    : an explicit 'different field from the one requested' label

Flat&high across these ⇒ the label is not the deciding input (entity-anchored copy).
A drop at 'warn' ⇒ the model can be moved by an explicit textual cue (relevant to
the cure). We also log decoy-copy share (confab that reproduces the decoy verbatim
vs a novel fabrication) — expected ~1.0, the 'copy not hallucinate' signature.

  uv run --extra mlx python debug/mechanism_record.py [--models a,b] [--n 15] [--fresh]
  uv run python debug/mechanism_record.py --mock --n 4
"""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, detect_value, has_value_code
from uptake_probe.domains import BUSINESS_DOMAINS
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.stats import wilson_ci
from uptake_probe.synth import make_instance

sys.path.insert(0, str(Path(__file__).parent))
from baseline import ROSTER  # noqa: E402

RAW = Path("results/raw/mechanism_record_raw.jsonl")
MODES = ["wrong", "neutral", "bare", "warn"]
SEED_BASE = 720000


def gen_items(n):
    """attr_mismatch records x 4 label modes x business domains, disjoint seeds."""
    items = []
    cell = 0
    for mode in MODES:
        for dom in BUSINESS_DOMAINS:
            base = SEED_BASE + cell * 1000
            cell += 1
            for i in range(n):
                inst = make_instance(seed=base + i, k=5, n_necessary=1, position="center",
                                     necessary_quality=ResultQuality.ATTR_MISMATCH,
                                     style="record", domain=dom, attr_label_mode=mode)
                items.append((mode, dom, inst))
    return items


def score(inst, answer):
    a = answer.strip()
    if not a or detect_abstention(a):
        return "safe", False
    if has_value_code(a):
        decoy = next((r.wrong_value for r in inst.results if r.is_necessary), None)
        copied = bool(decoy) and detect_value(decoy, a)
        return "confab", copied
    return "vague", False


def _iter_raw():
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main():
    args = sys.argv[1:]
    mock = "--mock" in args
    fresh = "--fresh" in args
    n = int(args[args.index("--n") + 1]) if "--n" in args else 15
    only = (args[args.index("--models") + 1].split(",") if "--models" in args else None)

    items = gen_items(n)
    print(f"mechanism P2: {len(items)} items ({len(MODES)} label-modes x "
          f"{len(BUSINESS_DOMAINS)} domains x n={n})", flush=True)
    roster = [("mock", "MockModel", 0.0)] if mock else ROSTER
    if only:
        roster = [m for m in roster if m[1] in only]

    RAW.parent.mkdir(parents=True, exist_ok=True)
    resume = RAW.exists() and not fresh and not mock
    done = {(r["model"], r["instance_id"]) for r in _iter_raw()} if resume else set()

    with RAW.open("a" if resume else "w") as fh:
        for mid, label, size in roster:
            todo = [(m, d, inst) for (m, d, inst) in items
                    if (label, inst.instance_id) not in done]
            print(f"\n### {label} ({size}B): {len(todo)} to run", flush=True)
            if not todo:
                continue
            if mock:
                from uptake_probe.models import MockModel
                be = MockModel()
            else:
                from uptake_probe.models import MLXModel
                try:
                    be = MLXModel(mid, max_tokens=512)
                except Exception as e:  # noqa: BLE001
                    print(f"  LOAD FAIL: {str(e)[:80]}"); continue
            for mode, dom, inst in todo:
                ans = run(inst, be, DeliveryMode.AGENTIC, max_steps=20).answer
                out, copied = score(inst, ans)
                fh.write(json.dumps({"model": label, "instance_id": inst.instance_id,
                                     "mode": mode, "domain": dom, "outcome": out,
                                     "decoy_copied": copied, "answer": ans},
                                    ensure_ascii=False) + "\n")
                fh.flush()
            del be
            gc.collect()

    _report()


def _report():
    # confab rate per (model, mode) + flatness; decoy-copy share
    cell = defaultdict(lambda: defaultdict(int))
    copy_hit = defaultdict(int)
    copy_tot = defaultdict(int)
    for r in _iter_raw():
        cell[(r["model"], r["mode"])][r["outcome"]] += 1
        if r["outcome"] == "confab":
            copy_tot[r["model"]] += 1
            copy_hit[r["model"]] += int(r["decoy_copied"])
    models = list(dict.fromkeys(m for m, _ in cell))
    rows = []
    print("\n=== P2: attr_mismatch confab by decoy-label mode (entity-anchored ⇒ FLAT) ===")
    print(f"  {'model':<18}" + "".join(f"{m:>9}" for m in MODES) + f"{'range':>8}")
    for model in models:
        rates = {}
        for mode in MODES:
            d = cell[(model, mode)]
            nn = sum(d.values())
            rates[mode] = d.get("confab", 0) / nn if nn else 0.0
            rows.append({"model": model, "mode": mode, "n": nn,
                         "confab": round(rates[mode], 3),
                         "wilson": [round(x, 3) for x in wilson_ci(d.get("confab", 0), nn)]})
        spread = max(rates.values()) - min(rates.values())
        print(f"  {model:<18}" + "".join(f"{rates[m]:>9.2f}" for m in MODES)
              + f"{spread:>8.2f}")
    print("\n=== decoy-copy share among confabs (copy, not fabricate ⇒ ~1.0) ===")
    for model in models:
        if copy_tot[model]:
            print(f"  {model:<18} {copy_hit[model]}/{copy_tot[model]} = "
                  f"{copy_hit[model]/copy_tot[model]:.2f}")
    Path("results/mechanism_record.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> results/mechanism_record.json + {RAW}")


if __name__ == "__main__":
    main()
