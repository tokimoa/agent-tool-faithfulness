"""Independent second-annotator naturalness audit of the shipped realistic docs.

The build-time QC judge was gemma-4-26b. To cross-check that the released docs
read naturally WITHOUT relying on the same judge, we re-rate a stratified sample
with a DIFFERENT local model (default Qwen3.5-27B) and report the natural-rate and
any flagged docs. High agreement = the naturalness of the dataset is not an
artifact of one judge. (Still generation-side only; scoring stays judge-free.)

  uv run --extra mlx python debug/audit_naturalness.py [n_per_cell]
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.qc import judge_naturalness

DS = Path("benchmark/dataset.jsonl")
AUDITOR = "mlx-community/Qwen3.5-27B-4bit"   # != build QC judge (gemma-4-26b)


def main() -> None:
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4
    recs = [json.loads(l) for l in DS.read_text().splitlines() if l.strip()]
    realistic = [r for r in recs if r["style"] == "record"]

    # stratified sample: per (domain, condition)
    buckets = defaultdict(list)
    for r in realistic:
        buckets[(r["domain"], r["condition"])].append(r)
    rng = random.Random(0)
    sample = []
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        sample.extend(items[:per_cell])
    print(f"auditing {len(sample)} realistic docs with independent judge: {AUDITOR}", flush=True)

    from uptake_probe.models import MLXModel
    be = MLXModel(AUDITOR, max_tokens=64)
    natural = 0
    flagged = []
    by_dom = defaultdict(lambda: [0, 0])
    for r in sample:
        nec = next(x for x in r["results"] if x["is_necessary"])
        res = judge_naturalness(be, nec["full_text"])
        by_dom[r["domain"]][1] += 1
        if res.natural:
            natural += 1
            by_dom[r["domain"]][0] += 1
        else:
            flagged.append((r["domain"], r["condition"], nec["full_text"]))

    rate = natural / len(sample)
    out = {"auditor": AUDITOR, "n": len(sample), "natural_rate": round(rate, 3),
           "by_domain": {d: round(a / t, 3) for d, (a, t) in by_dom.items()},
           "n_flagged": len(flagged)}
    Path("results/audit_naturalness.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nindependent natural-rate: {rate:.3f}  ({natural}/{len(sample)})")
    for d, (a, t) in sorted(by_dom.items()):
        print(f"  {d:<10} {a}/{t} = {a/t:.2f}")
    if flagged:
        print(f"\n{len(flagged)} flagged as AWKWARD by the independent judge:")
        for d, c, txt in flagged[:12]:
            print(f"  [{d}/{c}] {txt[:100]}")
    else:
        print("\n✓ no docs flagged by the independent judge (naturalness is robust)")
    print("saved -> results/audit_naturalness.json")


if __name__ == "__main__":
    main()
