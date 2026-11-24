"""Mitigation experiment — the 'cure' half of the mechanism->cure causal chain.

If silent mis-grounding is ENTITY-ANCHORED value completion with the attribute
constraint under-enforced, then a cure must put the (entity, attribute) binding
back into the OUTPUT CONTRACT. We compare four report-time interventions on the
same items (judge-free confab scoring):

  orig        : no intervention (the shipped report task)
  cite        : MUST name the exact source FIELD LABEL and only report a value if
                that label equals the asked attribute, else 'not available'
                -> forces the field binding the theory says is missing
  verify      : free 'double-check, else not available' (NO field binding)
                -> the control: predicted NOT to help attribute (verifier shares
                   the same blind spot)
  consequence : 'a wrong value is worse than not available'
                -> lowers answer-completion pressure (a separate lever)

Predicted signature (the causal evidence):
  confab(cite) << confab(orig)   on attribute, but confab(verify) ~= confab(orig).
A cure that works BECAUSE it targets the mechanism — not prompt trial-and-error.

  uv run --extra mlx python debug/mitigation_contrast.py [--models a,b] [--n 12] [--fresh]
  uv run python debug/mitigation_contrast.py --mock --n 3
"""

from __future__ import annotations

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

RAW = Path("results/raw/mitigation_contrast_raw.jsonl")
Q = ResultQuality
DIMS = {"attr_mismatch": Q.ATTR_MISMATCH, "irrelevant": Q.IRRELEVANT, "stale": Q.STALE}
INTERVENTIONS = {
    "orig": "",
    "cite": ("IMPORTANT: For any value you report you MUST also name the EXACT field "
             "label you read it from, e.g. value AB12-CD34 from field \"booking "
             "reference\". Only report a value if that field label is EXACTLY the "
             "attribute requested in the question. If no field with that exact "
             "attribute is present, answer: not available."),
    "verify": ("IMPORTANT: Before answering, double-check your answer is correct and "
               "supported by the results. If you are not sure, answer: not available."),
    "consequence": ("IMPORTANT: Reporting a wrong or unverified value is worse than "
                    "saying you don't have it. If the exact requested value is not "
                    "clearly present, answer: not available."),
}
SEED_BASE = 740000
# cite-induced refusals the global detector doesn't catch (kept local to this probe)
_EXTRA_ABSTAIN = re.compile(
    r"not[_ ]available|no (matching|exact|such) field|different field|"
    r"does not match|doesn't match|field .* not present", re.I)


def gen_items(n):
    items = []
    cell = 0
    for dim, q in DIMS.items():
        for dom in BUSINESS_DOMAINS:
            base = SEED_BASE + cell * 1000
            cell += 1
            for i in range(n):
                inst = make_instance(seed=base + i, k=5, n_necessary=1, position="center",
                                     necessary_quality=q, style="record", domain=dom)
                items.append((dim, dom, inst))
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


def main():
    args = sys.argv[1:]
    mock = "--mock" in args
    fresh = "--fresh" in args
    n = int(args[args.index("--n") + 1]) if "--n" in args else 12
    only = (args[args.index("--models") + 1].split(",") if "--models" in args else None)

    items = gen_items(n)
    total = len(items) * len(INTERVENTIONS)
    print(f"mitigation: {len(items)} items x {len(INTERVENTIONS)} interventions = "
          f"{total} runs/model", flush=True)
    roster = [("mock", "MockModel", 0.0)] if mock else ROSTER
    if only:
        roster = [m for m in roster if m[1] in only]

    RAW.parent.mkdir(parents=True, exist_ok=True)
    resume = RAW.exists() and not fresh and not mock
    done = {(r["model"], r["instance_id"], r["interv"]) for r in _iter_raw()} if resume else set()

    with RAW.open("a" if resume else "w") as fh:
        for mid, label, size in roster:
            todo = [(dim, dom, inst, iv) for (dim, dom, inst) in items
                    for iv in INTERVENTIONS if (label, inst.instance_id, iv) not in done]
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
            for dim, dom, inst, iv in todo:
                ans = run(inst, be, DeliveryMode.AGENTIC, max_steps=20,
                          extra=INTERVENTIONS[iv]).answer
                fh.write(json.dumps({"model": label, "instance_id": inst.instance_id,
                                     "dim": dim, "domain": dom, "interv": iv,
                                     "outcome": score(ans), "answer": ans},
                                    ensure_ascii=False) + "\n")
                fh.flush()
            del be
            gc.collect()

    _report()


def _report():
    cell = defaultdict(lambda: defaultdict(int))  # (model,dim,interv)->outcome counts
    for r in _iter_raw():
        cell[(r["model"], r["dim"], r["interv"])][r["outcome"]] += 1
    models = list(dict.fromkeys(k[0] for k in cell))
    ivs = list(INTERVENTIONS)
    rows = []
    for dim in DIMS:
        print(f"\n=== {dim}: confab rate by intervention "
              f"({'cite should DROP, verify should NOT' if dim=='attr_mismatch' else 'cross-dim'}) ===")
        print(f"  {'model':<18}" + "".join(f"{iv:>13}" for iv in ivs))
        for model in models:
            cells = []
            for iv in ivs:
                d = cell[(model, dim, iv)]
                nn = sum(d.values())
                rate = d.get("confab", 0) / nn if nn else 0.0
                lo, hi = wilson_ci(d.get("confab", 0), nn)
                cells.append(f"{rate:.2f}")
                rows.append({"model": model, "dim": dim, "interv": iv, "n": nn,
                             "confab": round(rate, 3), "wilson": [round(lo, 3), round(hi, 3)]})
            print(f"  {model:<18}" + "".join(f"{c:>13}" for c in cells))
    # headline summary: attribute cite-vs-orig and verify-vs-orig deltas, pooled
    print("\n=== headline (attribute, pooled over models): Δconfab vs orig ===")
    for iv in ["cite", "verify", "consequence"]:
        o = sum(cell[(m, 'attr_mismatch', 'orig')].get('confab', 0) for m in models)
        on = sum(sum(cell[(m, 'attr_mismatch', 'orig')].values()) for m in models)
        x = sum(cell[(m, 'attr_mismatch', iv)].get('confab', 0) for m in models)
        xn = sum(sum(cell[(m, 'attr_mismatch', iv)].values()) for m in models)
        if on and xn:
            print(f"  {iv:<12} {x/xn:.2f} vs orig {o/on:.2f}  (Δ {x/xn - o/on:+.2f})")
    Path("results/mitigation_contrast.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> results/mitigation_contrast.json + {RAW}")


if __name__ == "__main__":
    main()
