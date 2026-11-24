"""Human-validation harness for the judge-free detector (reduces the "your regex
classifier might be wrong — you found 8 bugs in it" criticism).

Two steps:
  1) sample : draw a stratified sample from results/raw/taxonomy_raw.jsonl
              (balanced across conditions and auto_confab T/F) and write a blind
              labeling sheet results/detector_validation_sheet.jsonl, each row with
              a blank "human_confab" to fill (true/false) from {asked, decoy, answer}.
  2) score  : read the labeled sheet, report agreement (accuracy + Cohen's kappa +
              confusion) between the classifier (auto_confab) and the human label.

  uv run python debug/validate_detector.py sample [N]
  uv run python debug/validate_detector.py score
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("results/raw/taxonomy_raw.jsonl")
SHEET = Path("results/detector_validation_sheet.jsonl")


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample(n: int = 160) -> None:
    rows = _load(RAW)
    # stratify by (cond, auto_confab) so both classes and all conditions are covered
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["cond"], bool(r["auto_confab"]))].append(r)
    rng = random.Random(0)
    per = max(1, n // max(1, len(strata)))
    picked = []
    for key, items in sorted(strata.items()):
        rng.shuffle(items)
        picked.extend(items[:per])
    rng.shuffle(picked)
    with SHEET.open("w") as fh:
        for i, r in enumerate(picked):
            fh.write(json.dumps({
                "id": i,
                "model": r["model"], "cond": r["cond"],
                "asked": f"{r['asked_attribute']} for {r['asked_entity']}",
                "decoy": r["decoy"], "answer": r["answer"],
                "auto_confab": bool(r["auto_confab"]),
                "human_confab": None,   # <- FILL: true if the answer asserts a value as THE asked answer
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} items across {len(strata)} strata -> {SHEET}")
    print("Label each row's human_confab (true/false), then: python debug/validate_detector.py score")


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def score() -> None:
    rows = _load(SHEET)
    labeled = [r for r in rows if r.get("human_confab") is not None]
    if not labeled:
        print("no human_confab labels filled yet."); return
    auto = [bool(r["auto_confab"]) for r in labeled]
    human = [bool(r["human_confab"]) for r in labeled]
    acc = sum(1 for x, y in zip(auto, human) if x == y) / len(labeled)
    conf = Counter((a, h) for a, h in zip(auto, human))
    print(f"labeled: {len(labeled)}/{len(rows)}")
    print(f"accuracy (auto vs human): {acc:.3f}")
    print(f"Cohen's kappa: {_kappa(auto, human):.3f}")
    print("confusion (auto, human): "
          f"TT={conf[(True, True)]} TF={conf[(True, False)]} "
          f"FT={conf[(False, True)]} FF={conf[(False, False)]}")
    disagree = [r for r in labeled if bool(r["auto_confab"]) != bool(r["human_confab"])]
    if disagree:
        print(f"\n{len(disagree)} disagreements (inspect these):")
        for r in disagree[:20]:
            print(f"  [{r['cond']}] auto={r['auto_confab']} human={r['human_confab']} "
                  f"decoy={r['decoy']} | {r['answer'][:90]!r}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if cmd == "sample":
        n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 160
        sample(n)
    elif cmd == "score":
        score()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
