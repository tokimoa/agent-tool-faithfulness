"""Input-ladder generalization: entity (irrelevant) and recency (stale) rungs,
plus a GOOD-side over-abstention control.

The paper's input ladder (input_cure.py) is attribute-only. Reviewers ask two
things: (1) do the input rungs generalise to the dimensions where the WHOLE
record (not one field) is wrong, and (2) do the rungs cost anything on GOOD
results. This script answers both, judge-free, on the record-style instrument:

  irrelevant (entity)   orig     : wrong-entity record shipped as-is
                        absence  : + explicit 'no record for <asked entity>'
                                   line (the wrong-entity record stays)  [salience]
                        isolate  : entity-scoped tool -> 'no record on file'
                                   (the wrong-entity record is REMOVED)  [structural]
  stale (recency)       orig     : superseded value + in-document marker
                        absence  : + explicit 'current <field>: no value on file'
                                   line (the superseded value stays)     [salience]
                        isolate  : recency-scoped tool -> only the current value
                                   is returned, i.e. none (old value REMOVED)
  good (control)        orig     : shipped record
                        isolate  : tool returns only the asked field (with the
                                   gold value) -- utilization must survive

Every raw row also records code_present (any complete code in the answer),
because on stale the mis-grounding rate measures caveating; the structural rung
should kill code_present too.

  uv run --extra mlx python debug/ladder_dims.py [--models a,b] [--n 15] [--fresh]
  uv run python debug/ladder_dims.py --mock --n 2
  uv run python debug/ladder_dims.py --show      # offline: print the doc variants
"""

from __future__ import annotations

import dataclasses
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, fact_in_text, has_value_code
from uptake_probe.domains import BUSINESS_DOMAINS
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.stats import wilson_ci
from uptake_probe.synth import make_instance

sys.path.insert(0, str(Path(__file__).parent))
from baseline import ROSTER  # noqa: E402
from mitigation_contrast import _EXTRA_ABSTAIN  # noqa: E402

RAW = Path("results/raw/ladder_dims_raw.jsonl")
OUT = Path("results/ladder_dims.json")
SEED_BASE = 810000

CORE6 = ["Phi-4-mini", "aya-expanse-8b", "gemma-3-12b",
         "gemma-4-26b", "Qwen3.5-27B", "Qwen3.6-35B"]

# Echoes of the injected interface phrasings count as abstention (abstention
# priority; consistent with detect.py's "not available/not recorded" family --
# these are new phrasings the frozen regex predates).
_LADDER_ABSTAIN = re.compile(
    r"no (?:value|record)s? on file|not present in this record"
    r"|not covered by this record|no current value|superseded entry", re.I)

GRID = {
    "irrelevant": (ResultQuality.IRRELEVANT, ["orig", "absence", "isolate"]),
    "stale": (ResultQuality.STALE, ["orig", "absence", "isolate"]),
    "good": (ResultQuality.GOOD, ["orig", "isolate"]),
}


def _nec(inst):
    return next(r for r in inst.results if r.is_necessary)


def _asked_entity(inst):
    if inst.unobtainable_entities:
        return next(iter(inst.unobtainable_entities))
    return _nec(inst).fact.entity


def mutate(inst, cond, lever):
    """Rewrite the necessary result for an input lever (orig leaves it alone)."""
    if lever == "orig":
        return inst
    nec = _nec(inst)
    asked_attr = nec.fact.attribute
    asked_ent = _asked_entity(inst)
    header = nec.full_text.split("\n")[0]
    if cond == "irrelevant":
        if lever == "absence":
            new = nec.full_text + (f"\n  - requested entity '{asked_ent}': no record on "
                                   "file (this record concerns a different entity)")
        elif lever == "isolate":
            # entity-scoped tool: the lookup for the asked entity returns nothing;
            # the wrong-entity record is not co-presented at all.
            new = (f"[Registry] lookup for {asked_ent}"
                   f"\n  - requested entity '{asked_ent}': no record on file")
        else:
            raise ValueError(lever)
    elif cond == "stale":
        if lever == "absence":
            new = nec.full_text + (f"\n  - current {asked_attr}: no value on file "
                                   f"(the {asked_attr} above is a superseded entry)")
        elif lever == "isolate":
            # recency-scoped tool: only the current value of the asked field is
            # returned, i.e. none; the superseded value is removed.
            new = header + (f"\n  - requested field '{asked_attr}' (current): no value "
                            "on file in this record")
        else:
            raise ValueError(lever)
    elif cond == "good":
        if lever == "isolate":
            # field-scoped tool: only the asked field is returned -- with the gold
            # value. Utilization must survive the narrowing.
            new = header + f"\n  - {asked_attr}: {nec.fact.value}"
        else:
            raise ValueError(lever)
    else:
        raise ValueError(cond)
    nec2 = dataclasses.replace(nec, full_text=new)
    return dataclasses.replace(inst, results=[nec2 if r.is_necessary else r
                                              for r in inst.results])


def gen_items(n):
    """{cond: [instances]} -- distinct seed stride per (cond, domain) cell."""
    items = {}
    cell = 0
    for cond in GRID:
        quality = GRID[cond][0]
        out = []
        for dom in BUSINESS_DOMAINS:
            base = SEED_BASE + cell * 1000
            cell += 1
            for i in range(n):
                out.append(make_instance(seed=base + i, k=5, n_necessary=1,
                                         position="center", necessary_quality=quality,
                                         style="record", domain=dom))
        items[cond] = out
    return items


def score(cond, inst, answer):
    a = (answer or "").strip()
    code_present = has_value_code(a)
    abstains = (not a or detect_abstention(a) or _EXTRA_ABSTAIN.search(a)
                or _LADDER_ABSTAIN.search(a))
    if cond == "good":
        if fact_in_text(_nec(inst).fact, a):
            outcome = "used"
        elif abstains:
            outcome = "abstain"
        else:
            outcome = "miss"
    else:
        if abstains:
            outcome = "safe"
        else:
            outcome = "confab" if code_present else "vague"
    return outcome, code_present


def _iter_raw():
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def show():
    items = gen_items(1)
    for cond, (_, levers) in GRID.items():
        inst = items[cond][0]
        print(f"===== {cond} (asked entity: {_asked_entity(inst)!r}, "
              f"asked field: {_nec(inst).fact.attribute!r}, "
              f"gold: {_nec(inst).fact.value!r}) =====")
        for lever in levers:
            mi = mutate(inst, cond, lever)
            print(f"--- {lever} ---")
            print("  " + _nec(mi).full_text.replace("\n", "\n  "))
        print()


def main():
    args = sys.argv[1:]
    if "--show" in args:
        show(); return
    mock = "--mock" in args
    fresh = "--fresh" in args
    n = int(args[args.index("--n") + 1]) if "--n" in args else 15
    only = (args[args.index("--models") + 1].split(",") if "--models" in args else None)

    items = gen_items(n)
    total = sum(len(v) * len(GRID[c][1]) for c, v in items.items())
    print(f"ladder-dims: {total} runs/model "
          f"({', '.join(f'{c}x{len(GRID[c][1])}' for c in GRID)}; n={n}/domain)",
          flush=True)
    roster = ([("mock", "MockModel", 0.0)] if mock
              else [m for m in ROSTER if m[1] in (only or CORE6)])

    RAW.parent.mkdir(parents=True, exist_ok=True)
    resume = RAW.exists() and not fresh and not mock
    done = ({(r["model"], r["cond"], r["instance_id"], r["lever"]) for r in _iter_raw()}
            if resume else set())

    with RAW.open("a" if resume else "w") as fh:
        for mid, label, size in roster:
            todo = [(cond, inst, lev)
                    for cond, insts in items.items()
                    for inst in insts
                    for lev in GRID[cond][1]
                    if (label, cond, inst.instance_id, lev) not in done]
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
            for cond, inst, lever in todo:
                mi = mutate(inst, cond, lever)
                ans = run(mi, be, DeliveryMode.AGENTIC, max_steps=20).answer
                outcome, code_present = score(cond, inst, ans)
                fh.write(json.dumps({"model": label, "cond": cond,
                                     "instance_id": inst.instance_id, "lever": lever,
                                     "outcome": outcome, "code_present": code_present,
                                     "answer": ans}, ensure_ascii=False) + "\n")
                fh.flush()
            del be
            gc.collect()

    _report()


def _report():
    cell = defaultdict(lambda: defaultdict(int))
    codep = defaultdict(lambda: [0, 0])  # (model, cond, lever) -> [k_code, n]
    for r in _iter_raw():
        key = (r["model"], r["cond"], r["lever"])
        cell[key][r["outcome"]] += 1
        codep[key][0] += int(r.get("code_present", False))
        codep[key][1] += 1
    models = list(dict.fromkeys(k[0] for k in cell))
    rows = []
    for cond, (_, levers) in GRID.items():
        bad = "used" if cond == "good" else "confab"
        print(f"\n=== {cond}: {'utilization' if cond == 'good' else 'mis-grounding'} "
              "by lever ===")
        print(f"  {'model':<18}" + "".join(f"{lv:>20}" for lv in levers))
        for model in models:
            cells = []
            for lv in levers:
                d = cell[(model, cond, lv)]
                nn = sum(d.values())
                rate = d.get(bad, 0) / nn if nn else 0.0
                kc, nc = codep[(model, cond, lv)]
                cp = kc / nc if nc else 0.0
                lo, hi = wilson_ci(d.get(bad, 0), nn)
                rows.append({"model": model, "cond": cond, "lever": lv, "n": nn,
                             ("util" if cond == "good" else "confab"): round(rate, 3),
                             "wilson": [round(lo, 3), round(hi, 3)],
                             "code_present": round(cp, 3)})
                cells.append(f"{rate:.2f}/cp{cp:.2f}")
            print(f"  {model:<18}" + "".join(f"{c:>20}" for c in cells))
        print(f"  --- pooled ---")
        for lv in levers:
            k = sum(cell[(m, cond, lv)].get(bad, 0) for m in models)
            nn = sum(sum(cell[(m, cond, lv)].values()) for m in models)
            kc = sum(codep[(m, cond, lv)][0] for m in models)
            if nn:
                lo, hi = wilson_ci(k, nn)
                print(f"  {lv:<9} {k/nn:.2f} [{lo:.2f},{hi:.2f}]  code_present {kc/nn:.2f}")
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {OUT} + {RAW}")


if __name__ == "__main__":
    main()
