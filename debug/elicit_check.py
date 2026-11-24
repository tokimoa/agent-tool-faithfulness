"""Elicitation check (construct-validity smoke test).

Does the FINAL benchmark actually trigger the failures the experiments found? The
shipped docs (multi-field `record` style) differ from the experiment carriers
(single-sentence template/realistic), so before any full measurement we verify the
SAME phenomena still appear, on a deliberately SMALL sample (11 models x ~10 items).
This is a directional sanity check, NOT a precise rate (CIs are huge at this N).

We fix a handful of items per headline condition and check the EXPECTED direction:
  good          -> USE the value        (uptake control: should be high)
  attr_mismatch -> CONFAB               (attribute blind spot)
  irrelevant    -> CONFAB               (entity over-trust)
  stale         -> CONFAB               (recency over-trust)
  partial       -> ABSTAIN / no confab  (completeness control: should be ~0)

Report framing (AGENTIC) matches the original experiments. Prints actual answers so
the behaviour can be eyeballed.

  uv run --extra mlx python debug/elicit_check.py [--per 2] [--models a,b]
  uv run python debug/elicit_check.py --mock --per 2
"""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode

sys.path.insert(0, str(Path(__file__).parent))
from baseline import ROSTER, _instance, score  # noqa: E402

DS = Path("benchmark/dataset.jsonl")
# (condition, headline dimension, expected direction)
PLAN = [
    ("good", "uptake-control", "USE"),
    ("attr_mismatch", "attribute", "CONFAB"),
    ("irrelevant", "entity", "CONFAB"),
    ("stale", "recency", "CONFAB"),
    ("partial", "completeness", "SAFE"),
]
_HARM = {"confab"}
_SAFE = {"abstain_silent", "abstain_explicit", "vague"}


def pick_items(recs, per):
    """`per` record-style items per planned condition, spread across domains."""
    by_cond = defaultdict(list)
    for r in recs:
        if r["style"] == "record":
            by_cond[r["condition"]].append(r)
    items = []
    for cond, _dim, _exp in PLAN:
        pool = sorted(by_cond.get(cond, []), key=lambda r: r["instance_id"])
        # spread across domains: round-robin by domain
        bydom = defaultdict(list)
        for r in pool:
            bydom[r["domain"]].append(r)
        doms = sorted(bydom)
        chosen, i = [], 0
        while len(chosen) < per and doms:
            d = doms[i % len(doms)]
            if bydom[d]:
                chosen.append(bydom[d].pop(0))
            else:
                doms.remove(d)
                continue
            i += 1
        items.extend(chosen)
    return items


def outcome_label(rec, out):
    if rec["unobtainable_entities"]:
        return "CONFAB" if out in _HARM else ("SAFE" if out in _SAFE else out.upper())
    return "USE" if out == "used" else "MISS"


def main():
    args = sys.argv[1:]
    mock = "--mock" in args
    per = int(args[args.index("--per") + 1]) if "--per" in args else 2
    only = (args[args.index("--models") + 1].split(",") if "--models" in args else None)

    recs = [json.loads(l) for l in DS.read_text().splitlines() if l.strip()]
    items = pick_items(recs, per)
    print(f"=== elicitation check: {len(items)} items x "
          f"{len(only) if only else (1 if mock else len(ROSTER))} models "
          f"(report framing) ===\n")
    print("selected items:")
    for r in items:
        q = r["question"].split("?")[0].split(")")[-1].strip() + "?"
        print(f"  [{r['condition']:14}/{r['domain']:9}] {q}  gold={r['gold_values']} "
              f"decoy={r['decoys']}")
    print()

    roster = [("mock", "MockModel", 0.0)] if mock else ROSTER
    if only:
        roster = [m for m in roster if m[1] in only]

    # results[model][condition] = list of (outcome_label, answer)
    results = defaultdict(lambda: defaultdict(list))
    examples = []
    for mid, label, size in roster:
        print(f"### {label} ({size}B)", flush=True)
        if mock:
            from uptake_probe.models import MockModel
            backend = MockModel()
        else:
            from uptake_probe.models import MLXModel
            try:
                backend = MLXModel(mid, max_tokens=512)
            except Exception as e:  # noqa: BLE001
                print(f"  LOAD FAIL: {str(e)[:80]} -- skip"); continue
        for r in items:
            inst = _instance(r)
            ans = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20).answer
            out = score(r, ans)
            lab = outcome_label(r, out)
            results[label][r["condition"]].append(lab)
            if len(examples) < 60:
                examples.append((label, r["condition"], r["domain"], r["gold_values"],
                                 r["decoys"], lab, ans[:80]))
        del backend
        gc.collect()

    _report(results, examples, [m[1] for m in roster])


def _report(results, examples, models):
    print("\n=== per-model x dimension (outcome counts) ===")
    hdr = f"{'model':<20}" + "".join(f"{dim[:10]:>13}" for _c, dim, _e in PLAN)
    print(hdr)
    for m in models:
        if m not in results:
            continue
        cells = []
        for cond, _dim, exp in PLAN:
            labs = results[m].get(cond, [])
            n = len(labs)
            key = "CONFAB" if exp == "CONFAB" else ("USE" if exp == "USE" else "SAFE")
            hit = sum(1 for x in labs if x == key)
            cells.append(f"{hit}/{n} {key[:4]}")
        print(f"{m:<20}" + "".join(f"{c:>13}" for c in cells))

    print("\n=== benchmark-level elicitation (does the intended failure appear?) ===")
    for cond, dim, exp in PLAN:
        labs = [x for m in results for x in results[m].get(cond, [])]
        n = len(labs) or 1
        if exp == "CONFAB":
            rate = sum(1 for x in labs if x == "CONFAB") / n
            verdict = "✓ elicited" if rate > 0.15 else "✗ NOT elicited"
            print(f"  {dim:<14} confab-rate={rate:.0%}  (expect HIGH) -> {verdict}")
        elif exp == "USE":
            rate = sum(1 for x in labs if x == "USE") / n
            verdict = "✓ uptake works" if rate > 0.5 else "✗ low uptake"
            print(f"  {dim:<14} use-rate   ={rate:.0%}  (expect HIGH) -> {verdict}")
        else:  # SAFE
            rate = sum(1 for x in labs if x == "CONFAB") / n
            verdict = "✓ control holds" if rate < 0.15 else "✗ unexpected confab"
            print(f"  {dim:<14} confab-rate={rate:.0%}  (expect ~0)  -> {verdict}")

    print("\n=== sample answers (eyeball that the benchmark elicits the behaviour) ===")
    for cond, _dim, _e in PLAN:
        shown = [e for e in examples if e[1] == cond][:3]
        for label, c, dom, gold, decoy, lab, ans in shown:
            print(f"  [{c}/{dom}] {label:<16} -> {lab:<7} | gold={gold} decoy={decoy}")
            print(f"      answer: {ans!r}")

    Path("results/elicit_check.json").write_text(json.dumps(
        {m: {c: results[m][c] for c in results[m]} for m in results},
        indent=2, ensure_ascii=False))
    print("\nsaved -> results/elicit_check.json")


if __name__ == "__main__":
    main()
