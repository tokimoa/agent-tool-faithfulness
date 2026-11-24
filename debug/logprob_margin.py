"""Internal evidence — the abstention margin behind the behavioural ladder.

The behavioural deep-dive shows silent mis-grounding is entity-anchored value
COPY, cured not by prompting but by changing the INPUT (input_cure.py: confab
collapses orig->absence->isolate). This probe asks the same question one layer
DOWN, with no sampling: at the moment of answering, what does the model PREFER —
committing the on-page decoy value, or abstaining?

For each attr_mismatch record we score two teacher-forced continuations after the
shared prefix '...\nANSWER:':
    decoy    : the planted wrong-field code   (the mis-grounded commit)
    abstain  : 'not available'                (the safe answer)
margin = mean_logp(abstain) - mean_logp(decoy)
  margin < 0  -> the model internally PREFERS committing the decoy  (confab prior)
  margin > 0  -> the model internally PREFERS abstaining            (safe)

We read this under the ladder's decisive rungs (reusing input_cure.mutate):
    orig      : decoy present                 -> predict margin < 0
    absence   : + 'requested field not present'
    isolate   : decoy removed ('no value on file') -> predict margin flips > 0
If the margin tracks the behavioural confab across models AND flips sign under the
input cures, the cure's mechanism is visible in the logits, not just the samples.

  uv run --extra mlx python debug/logprob_margin.py [--models a,b] [--n 8] [--fresh]
  uv run python debug/logprob_margin.py --mock --n 4      # CPU plumbing self-test
"""

from __future__ import annotations

import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.domains import BUSINESS_DOMAINS
from uptake_probe.scaffolds import _PASSIVE_HEADER
from uptake_probe.schema import ResultQuality
from uptake_probe.stats import wilson_ci
from uptake_probe.synth import make_instance
from uptake_probe.tools import ToolEnv

sys.path.insert(0, str(Path(__file__).parent))
from baseline import ROSTER  # noqa: E402
from input_cure import mutate  # reuse the exact ladder rungs  # noqa: E402

RAW = Path("results/raw/logprob_margin_raw.jsonl")
CODE = re.compile(r"[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}")
RUNGS = ["orig", "absence", "isolate"]
ABSTAIN = "not available"
SEED_BASE = 780000


def _decoy(inst):
    return next((r.wrong_value for r in inst.results if r.is_necessary), None)


def _prefix(inst):
    """The passive answer-point: question + retrieved record(s), ending at ANSWER:."""
    block = ToolEnv(inst).passive_block()
    return (f"{_PASSIVE_HEADER}\nQUESTION: {inst.question}\n\n"
            f"RETRIEVED RESULTS:\n{block}\nANSWER:")


def gen_items(n):
    items = []
    cell = 0
    for dom in BUSINESS_DOMAINS:
        base = SEED_BASE + cell * 1000
        cell += 1
        for i in range(n):
            items.append(make_instance(seed=base + i, k=5, n_necessary=1,
                                       position="center",
                                       necessary_quality=ResultQuality.ATTR_MISMATCH,
                                       style="record", domain=dom))
    return items


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
    n = int(args[args.index("--n") + 1]) if "--n" in args else 8
    only = (args[args.index("--models") + 1].split(",") if "--models" in args else None)

    items = gen_items(n)
    print(f"logprob margin: {len(items)} attr_mismatch items x {len(RUNGS)} rungs "
          f"= {len(items) * len(RUNGS)} scored pairs/model", flush=True)
    roster = [("mock", "MockModel", 0.0)] if mock else ROSTER
    if only:
        roster = [m for m in roster if m[1] in only]

    RAW.parent.mkdir(parents=True, exist_ok=True)
    resume = RAW.exists() and not fresh and not mock
    done = {(r["model"], r["instance_id"], r["rung"]) for r in _iter_raw()} if resume else set()

    with RAW.open("a" if resume else "w") as fh:
        for mid, label, size in roster:
            todo = [(inst, rung) for inst in items for rung in RUNGS
                    if (label, inst.instance_id, rung) not in done]
            print(f"\n### {label} ({size}B): {len(todo)} pairs to score", flush=True)
            if not todo:
                continue
            if mock:
                from uptake_probe.models import MockModel
                be = MockModel()
            else:
                from uptake_probe.models import MLXModel
                try:
                    be = MLXModel(mid, max_tokens=8)
                except Exception as e:  # noqa: BLE001
                    print(f"  LOAD FAIL: {str(e)[:80]}"); continue
            for inst, rung in todo:
                decoy = _decoy(inst)
                if not decoy:
                    continue
                mi = mutate(inst, rung)
                prefix = _prefix(mi)
                lp_decoy = be.logprob(prefix, " " + decoy)
                lp_abstain = be.logprob(prefix, " " + ABSTAIN)
                fh.write(json.dumps({
                    "model": label, "instance_id": inst.instance_id, "rung": rung,
                    "lp_decoy": round(lp_decoy, 4), "lp_abstain": round(lp_abstain, 4),
                    "margin": round(lp_abstain - lp_decoy, 4)}, ensure_ascii=False) + "\n")
                fh.flush()
            del be
            gc.collect()

    _report()


def _report():
    cell = defaultdict(list)            # (model,rung) -> [margins]
    for r in _iter_raw():
        cell[(r["model"], r["rung"])].append(r["margin"])
    models = list(dict.fromkeys(k[0] for k in cell))
    rows = []
    print("\n=== abstention margin = logp(abstain) - logp(decoy)  "
          "(orig<0 => commits decoy; isolate>0 => abstains) ===")
    print(f"  {'model':<18}" + "".join(f"{r:>10}" for r in RUNGS) + f"{'flip?':>14}")
    for model in models:
        cells, mean = [], {}
        for rung in RUNGS:
            ms = cell[(model, rung)]
            mean[rung] = sum(ms) / len(ms) if ms else float("nan")
            neg = sum(1 for x in ms if x < 0)
            lo, hi = wilson_ci(neg, len(ms)) if ms else (0, 0)
            cells.append(f"{mean[rung]:+.2f}")
            rows.append({"model": model, "rung": rung, "n": len(ms),
                         "mean_margin": round(mean[rung], 3),
                         "frac_commit_decoy": round(neg / len(ms), 3) if ms else None,
                         "wilson_commit": [round(lo, 3), round(hi, 3)]})
        flip = (f"{mean['orig']:+.2f}->{mean['isolate']:+.2f}"
                if mean.get("orig") == mean.get("orig") else "n/a")
        print(f"  {model:<18}" + "".join(f"{c:>10}" for c in cells) + f"{flip:>14}")
    print("\n=== pooled mean margin by rung ===")
    for rung in RUNGS:
        allm = [x for m in models for x in cell[(m, rung)]]
        if allm:
            neg = sum(1 for x in allm if x < 0)
            print(f"  {rung:<9} {sum(allm)/len(allm):+.2f}   "
                  f"commit-decoy {neg}/{len(allm)} = {neg/len(allm):.2f}")
    Path("results/logprob_margin.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> results/logprob_margin.json + {RAW}")


if __name__ == "__main__":
    main()
