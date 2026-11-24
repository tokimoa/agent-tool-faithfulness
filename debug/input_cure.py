"""Input-level cures — locating WHERE the fix has to live.

Prompt-level cures (mitigation_contrast.py) and external supervision (a verifier
agent, which shares the blind spot) are weak. So we climb a LADDER from "ask the
model" to "change what the model sees", on the sticky `attr_mismatch` dimension,
and read off where confab finally collapses:

  orig     : do nothing (shipped record)                         [baseline]
  cite     : PROMPT — make the model name+match the source field [prompt-level]
  absence  : INPUT — add an explicit 'requested field NOT present' line
             (the decoy is still there; the model must heed the marker)  [salience]
  tags     : INPUT — tag every code field '[does NOT match requested field]'
             (the environment pre-computes the per-field match)          [salience]
  isolate  : INPUT — the tool returns only the asked field, i.e. 'no value
             on file' (the wrong-field value is REMOVED)                 [structural]

Reading: if confab only collapses at `isolate`, the cure is STRUCTURAL (fix the
tool/retrieval interface — don't co-present plausible wrong-field values), not
promptable. If `absence`/`tags` already fix it, an explicit input signal suffices.
If even `cite`/`absence` are ignored, the blindness is deep (cf. lexical_signal).
Judge-free confab scoring throughout.

  uv run --extra mlx python debug/input_cure.py [--models a,b] [--n 15] [--fresh]
  uv run python debug/input_cure.py --mock --n 3
  uv run python debug/input_cure.py --show          # offline: print the 5 doc variants
"""

from __future__ import annotations

import dataclasses
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, has_value_code
from uptake_probe.domains import BUSINESS_DOMAINS
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.stats import wilson_ci
from uptake_probe.synth import make_instance

sys.path.insert(0, str(Path(__file__).parent))
from baseline import ROSTER  # noqa: E402
from mitigation_contrast import INTERVENTIONS, _EXTRA_ABSTAIN  # reuse cite + abstain  # noqa: E402

RAW = Path("results/raw/input_cure_raw.jsonl")
CODE = re.compile(r"[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}")
LEVERS = ["orig", "cite", "absence", "tags", "isolate"]
SEED_BASE = 760000


def _asked(inst):
    nec = next(r for r in inst.results if r.is_necessary)
    return nec.fact.attribute


def mutate(inst, lever):
    """Return an instance whose necessary result is rewritten for an INPUT lever
    (orig/cite are prompt levers and leave the doc unchanged)."""
    if lever in ("orig", "cite"):
        return inst
    nec = next(r for r in inst.results if r.is_necessary)
    asked = nec.fact.attribute
    lines = nec.full_text.split("\n")
    header = lines[0]
    if lever == "absence":
        new = nec.full_text + (f"\n  - requested field '{asked}': not present in this "
                               "record")
    elif lever == "tags":
        out = []
        for ln in lines:
            if CODE.search(ln) and ln.lstrip().startswith("-"):
                out.append(ln + f"  [does NOT match requested field '{asked}']")
            else:
                out.append(ln)
        new = "\n".join(out)
    elif lever == "isolate":
        # field-scoped tool: only the asked field is returned -> not on file
        new = header + (f"\n  - requested field '{asked}': no value on file in this "
                        "record")
    else:
        raise ValueError(lever)
    nec2 = dataclasses.replace(nec, full_text=new)
    return dataclasses.replace(inst, results=[nec2 if r.is_necessary else r
                                              for r in inst.results])


def gen_items(n):
    items = []
    cell = 0
    for dom in BUSINESS_DOMAINS:
        base = SEED_BASE + cell * 1000
        cell += 1
        for i in range(n):
            items.append(make_instance(seed=base + i, k=5, n_necessary=1, position="center",
                                       necessary_quality=ResultQuality.ATTR_MISMATCH,
                                       style="record", domain=dom))
    return items


def score(answer):
    a = answer.strip()
    if not a or detect_abstention(a) or _EXTRA_ABSTAIN.search(a):
        return "safe"
    return "confab" if has_value_code(a) else "vague"


def _iter_raw():
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def show():
    inst = gen_items(1)[0]
    print(f"asked attribute: {_asked(inst)!r}\n")
    for lever in LEVERS:
        mi = mutate(inst, lever)
        nec = next(r for r in mi.results if r.is_necessary)
        extra = "  (+ prompt: cite-the-field)" if lever == "cite" else ""
        print(f"--- {lever}{extra} ---")
        print("  " + nec.full_text.replace("\n", "\n  "))
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
    print(f"input-cure ladder: {len(items)} attr_mismatch items x {len(LEVERS)} levers "
          f"= {len(items) * len(LEVERS)} runs/model", flush=True)
    roster = [("mock", "MockModel", 0.0)] if mock else ROSTER
    if only:
        roster = [m for m in roster if m[1] in only]

    RAW.parent.mkdir(parents=True, exist_ok=True)
    resume = RAW.exists() and not fresh and not mock
    done = {(r["model"], r["instance_id"], r["lever"]) for r in _iter_raw()} if resume else set()

    with RAW.open("a" if resume else "w") as fh:
        for mid, label, size in roster:
            todo = [(inst, lev) for inst in items for lev in LEVERS
                    if (label, inst.instance_id, lev) not in done]
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
            for inst, lever in todo:
                mi = mutate(inst, lever)
                extra = INTERVENTIONS["cite"] if lever == "cite" else ""
                ans = run(mi, be, DeliveryMode.AGENTIC, max_steps=20, extra=extra).answer
                fh.write(json.dumps({"model": label, "instance_id": inst.instance_id,
                                     "lever": lever, "outcome": score(ans),
                                     "answer": ans}, ensure_ascii=False) + "\n")
                fh.flush()
            del be
            gc.collect()

    _report()


def _report():
    cell = defaultdict(lambda: defaultdict(int))
    for r in _iter_raw():
        cell[(r["model"], r["lever"])][r["outcome"]] += 1
    models = list(dict.fromkeys(k[0] for k in cell))
    rows = []
    print("\n=== attr_mismatch confab along the cure ladder "
          "(where does it collapse?) ===")
    print(f"  {'model':<18}" + "".join(f"{lv:>10}" for lv in LEVERS))
    for model in models:
        cells = []
        for lv in LEVERS:
            d = cell[(model, lv)]
            nn = sum(d.values())
            rate = d.get("confab", 0) / nn if nn else 0.0
            cells.append(f"{rate:.2f}")
            lo, hi = wilson_ci(d.get("confab", 0), nn)
            rows.append({"model": model, "lever": lv, "n": nn, "confab": round(rate, 3),
                         "wilson": [round(lo, 3), round(hi, 3)]})
        print(f"  {model:<18}" + "".join(f"{c:>10}" for c in cells))
    print("\n=== pooled confab by lever (the ladder) ===")
    for lv in LEVERS:
        k = sum(cell[(m, lv)].get("confab", 0) for m in models)
        nn = sum(sum(cell[(m, lv)].values()) for m in models)
        if nn:
            lo, hi = wilson_ci(k, nn)
            tag = {"orig": "do-nothing", "cite": "PROMPT", "absence": "input/salience",
                   "tags": "input/salience", "isolate": "input/STRUCTURAL"}[lv]
            print(f"  {lv:<9} {k/nn:.2f}  [{lo:.2f},{hi:.2f}]  ({tag})")
    Path("results/input_cure.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> results/input_cure.json + {RAW}")


if __name__ == "__main__":
    main()
