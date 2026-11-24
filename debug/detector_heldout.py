"""Held-out validation of the FROZEN abstention detector (post-patch).

The paper's kappa=0.933 is a development figure: the regex was patched on the
same 154 items it was validated on. This script draws a FRESH stratified
sample from the record-era baseline raw (never seen by any patch round),
classifies it with the frozen detector as-is, and emits a blind labeling
sheet. After one author fills `human_confab` (true iff the answer asserts a
complete code as the value WITHOUT any abstention/caveat -- same rubric as the
original sheet), --score reports held-out accuracy / Cohen kappa / FN count.

  uv run python debug/detector_heldout.py --make [--n 150]
  # ... hand-label results/detector_heldout_sheet.jsonl (human_confab: true/false)
  uv run python debug/detector_heldout.py --score
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, has_value_code

RAW = Path("results/raw/baseline_raw.jsonl")
OLD_SHEET = Path("results/detector_validation_sheet.jsonl")
SHEET = Path("results/detector_heldout_sheet.jsonl")
TXT = Path("results/detector_heldout_sheet.txt")
SEED = 424242

# silent-mismatch conditions only -- where the confab/safe boundary lives
CONDS = ["irrelevant", "irrelevant_near", "irrelevant_far",
         "attr_mismatch", "stale", "partial"]


def auto_confab(answer: str) -> bool:
    a = (answer or "").strip()
    if not a or detect_abstention(a):
        return False
    return has_value_code(a)


def make(n_target: int) -> None:
    old_answers = set()
    if OLD_SHEET.exists():
        for line in OLD_SHEET.read_text().splitlines():
            if line.strip():
                old_answers.add(json.loads(line)["answer"].strip())

    buckets = defaultdict(list)
    for line in RAW.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["condition"] not in CONDS:
            continue
        a = (r["answer"] or "").strip()
        if not a or a in old_answers:
            continue
        # stratify on (condition, code-present, auto label): the informative
        # cells are code-bearing answers, where abstention detection decides.
        key = (r["condition"], has_value_code(a), auto_confab(a))
        buckets[key].append(r)

    rng = random.Random(SEED)
    # weight code-bearing strata 3x (they carry the decision boundary)
    keys = sorted(buckets)
    weights = {k: (3 if k[1] else 1) for k in keys}
    total_w = sum(weights[k] for k in keys)
    picked = []
    for k in keys:
        take = max(2, round(n_target * weights[k] / total_w))
        pool = buckets[k]
        rng.shuffle(pool)
        picked.extend(pool[:min(take, len(pool))])
    rng.shuffle(picked)
    picked = picked[:n_target]

    with SHEET.open("w") as fh, TXT.open("w") as tx:
        for i, r in enumerate(picked):
            row = {"id": i, "model": r["model"], "cond": r["condition"],
                   "instance_id": r["instance_id"], "answer": r["answer"],
                   "auto_confab": auto_confab(r["answer"]),
                   "human_confab": None}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            tx.write(f"[{i}] ({r['condition']})\n"
                     f"    {r['answer'].strip()[:400]}\n"
                     f"    human_confab: \n\n")
    dist = defaultdict(int)
    for r in picked:
        dist[(r["condition"], auto_confab(r["answer"]))] += 1
    print(f"wrote {len(picked)} items -> {SHEET} (+ {TXT} for reading)")
    print("strata (condition, auto_confab):")
    for k in sorted(dist):
        print(f"  {k}: {dist[k]}")
    print("\nLabel human_confab in the jsonl (true/false), blind to auto_confab; "
          "then run --score.")


def score() -> None:
    rows = [json.loads(l) for l in SHEET.read_text().splitlines() if l.strip()]
    labelled = [r for r in rows if r.get("human_confab") is not None]
    if not labelled:
        print("no human labels yet"); return
    n = len(labelled)
    agree = sum(r["auto_confab"] == r["human_confab"] for r in labelled)
    fn = [r for r in labelled if r["human_confab"] and not r["auto_confab"]]
    fp = [r for r in labelled if not r["human_confab"] and r["auto_confab"]]
    p_o = agree / n
    p_auto = sum(r["auto_confab"] for r in labelled) / n
    p_hum = sum(r["human_confab"] for r in labelled) / n
    p_e = p_auto * p_hum + (1 - p_auto) * (1 - p_hum)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0
    print(f"held-out: n={n}  accuracy={p_o:.3f}  kappa={kappa:.3f}  "
          f"FN={len(fn)}  FP={len(fp)}")
    for r in fn:
        print(f"  FN[{r['id']}] ({r['cond']}): {r['answer'][:120]}")
    for r in fp:
        print(f"  FP[{r['id']}] ({r['cond']}): {r['answer'][:120]}")


if __name__ == "__main__":
    if "--make" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 150
        make(n)
    elif "--score" in sys.argv:
        score()
    else:
        print(__doc__)
