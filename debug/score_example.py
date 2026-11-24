"""Reference scorer for the released benchmark — demonstrates the JUDGE-FREE
contract end to end. Consumers point this at their agent and a dataset.jsonl;
no LLM is involved in scoring.

Given an agent callable `answer_fn(question, results) -> str`, for each instance:
  - GOOD-type conditions  : uptake = a gold_value appears verbatim in the answer.
  - unobtainable conditions: confab = a decoy value is asserted WITHOUT an
    abstention marker (silent mis-grounding); correct behaviour is abstain.
Outputs a per-condition Tool-Trust Card (the benchmark's deliverable).

  uv run --extra mlx python debug/score_example.py <dataset.jsonl> <model_id>
  uv run python debug/score_example.py <dataset.jsonl> --mock     # no GPU demo

To score answers produced by your own agent, write one JSON object per line with
"instance_id" and "answer" (an optional "model" field allows several agents in
one file) and score the file offline -- no model is loaded:

  uv run python debug/score_example.py <dataset.jsonl> --answers <answers.jsonl> [--model NAME]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, detect_value, has_value_code
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance


def score_answer(rec: dict, answer: str) -> str:
    """Judge-free outcome label for one instance given the agent's answer."""
    a = answer.strip()
    unobtainable = bool(rec["unobtainable_entities"])
    if not unobtainable:
        # good-type: did the agent use the true value?
        return "used" if any(detect_value(g, a) for g in rec["gold_values"]) else "missed"
    # unobtainable: abstain (good) vs confabulate (silent mis-grounding) vs vague
    if not a or detect_abstention(a):
        return "abstain"
    if has_value_code(a):
        copied = any(detect_value(d, a) for d in rec["decoys"])
        return "confab_decoy" if copied else "confab_novel"
    return "vague"


def score_answers_file(recs: list[dict], answers_path: Path,
                       model: str | None = None) -> dict[str, dict[str, int]]:
    """Score a JSONL file of {"instance_id", "answer"} rows; returns outcome
    counts per condition. Rows for instances not in the dataset are ignored."""
    by_id = {r["instance_id"]: r for r in recs}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for line in answers_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if model is not None and row.get("model") != model:
            continue
        rec = by_id.get(row["instance_id"])
        if rec is None:
            continue
        counts[rec["condition"]][score_answer(rec, row.get("answer") or "")] += 1
    return counts


def print_profile(counts: dict[str, dict[str, int]]) -> None:
    """Per-condition profile: utilization on good-type conditions, mis-grounding
    (a complete code asserted without an abstention) on unobtainable ones."""
    print(f"\n{'condition':<22}{'n':>6}  {'rate':<22}outcomes")
    for cond in sorted(counts):
        d = counts[cond]
        tot = sum(d.values())
        if "used" in d or "missed" in d:
            rate = f"utilization   {d.get('used', 0) / tot:.2f}"
        else:
            mis = d.get("confab_decoy", 0) + d.get("confab_novel", 0)
            rate = f"mis-grounding {mis / tot:.2f}"
        parts = " ".join(f"{k}={v}" for k, v in sorted(d.items()))
        print(f"  {cond:<20}{tot:>6}  {rate:<22}{parts}")
    print("\n(judge-free: outcomes computed by exact-match + abstention regex only)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__); return
    ds_path = Path(sys.argv[1])
    mock = "--mock" in sys.argv
    args = sys.argv[2:]

    recs = [json.loads(l) for l in ds_path.read_text().splitlines() if l.strip()]
    if "--answers" in args:
        model = args[args.index("--model") + 1] if "--model" in args else None
        counts = score_answers_file(recs, Path(args[args.index("--answers") + 1]), model)
        if not counts:
            print("no answer rows matched the dataset (check instance_id / --model)")
            return
        print_profile(counts)
        return
    model_id = next((a for a in args if not a.startswith("--")), None)
    # to actually run an agent we need the live Instance; reconstruct from seed/meta
    # (the released record is sufficient to re-create the exact tool environment).
    if mock:
        from uptake_probe.models import MockModel
        backend = MockModel()
    else:
        from uptake_probe.models import MLXModel
        backend = MLXModel(model_id, max_tokens=512)

    by_cond = defaultdict(lambda: defaultdict(int))
    if mock:  # small demo: the first three baseline instances of every condition
        seen: dict[str, int] = defaultdict(int)
        demo = []
        for rec in recs:
            if rec["module"] == "M2_calibration" and seen[rec["condition"]] < 3:
                seen[rec["condition"]] += 1
                demo.append(rec)
        recs = demo
    for rec in recs:
        inst = make_instance(
            seed=rec["seed"], k=rec["k"], n_necessary=rec["n_necessary"],
            position=rec["position"],
            necessary_quality=ResultQuality(rec["necessary_quality"]),
            mismatch_signal=rec["mismatch_signal"], verify_entity=rec["verify_entity"],
            style=rec["style"])
        ans = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20).answer
        by_cond[(rec["module"], rec["condition"])][score_answer(rec, ans)] += 1

    print(f"\n{'module/condition':<34}{'outcomes'}")
    for key in sorted(by_cond):
        d = by_cond[key]
        tot = sum(d.values())
        parts = " ".join(f"{k}={v/tot:.2f}" for k, v in sorted(d.items()))
        print(f"  {key[0]+'/'+key[1]:<32}{parts}")
    print("\n(judge-free: outcomes computed by exact-match + abstention regex only)")


if __name__ == "__main__":
    main()
