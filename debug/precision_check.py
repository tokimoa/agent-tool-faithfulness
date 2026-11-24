"""Quantization spot check: does the 4-bit profile survive at higher precision?

Every number in the paper comes from 4-bit MLX checkpoints (disclosed in
Limitations). This runs ONE model at 4-bit and at a higher-precision checkpoint
on the headline record-style conditions plus the two label-ablation modes that
define label-(in)sensitivity, with matched fresh seeds, and reports per-cell
rates and deltas. Target model: gemma-3-12b (label-insensitive; if its profile
were a quantization artifact, this is where it would show).

  uv run --extra mlx python debug/precision_check.py [--n 15]
  uv run python debug/precision_check.py --mock --n 2
"""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, fact_in_text, has_value_code
from uptake_probe.domains import BUSINESS_DOMAINS
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.stats import wilson_ci
from uptake_probe.synth import make_instance

RAW = Path("results/raw/precision_check_raw.jsonl")
OUT = Path("results/precision_check.json")
SEED_BASE = 830000
Q = ResultQuality

# (pair_label, 4bit_id, higher-precision candidates tried in order)
PAIRS = [
    ("gemma-3-12b",
     "mlx-community/gemma-3-12b-it-qat-4bit",
     ["mlx-community/gemma-3-12b-it-8bit",
      "mlx-community/gemma-3-12b-it-bf16"]),
    ("Phi-4-mini",
     "mlx-community/Phi-4-mini-instruct-4bit",
     ["mlx-community/Phi-4-mini-instruct-8bit",
      "mlx-community/Phi-4-mini-instruct-bf16"]),
]

# (cell_label, quality, attr_label_mode)
CELLS = [
    ("good", Q.GOOD, None),
    ("irrelevant", Q.IRRELEVANT, None),
    ("attr_mismatch", Q.ATTR_MISMATCH, None),
    ("stale", Q.STALE, None),
    ("attr_label_neutral", Q.ATTR_MISMATCH, "neutral"),
    ("attr_label_warn", Q.ATTR_MISMATCH, "warn"),
]


def gen_items(n):
    items = defaultdict(list)
    for ci, (clabel, quality, mode) in enumerate(CELLS):
        for di, dom in enumerate(BUSINESS_DOMAINS):
            base = SEED_BASE + (ci * len(BUSINESS_DOMAINS) + di) * 1000
            for i in range(n):
                kwargs = dict(seed=base + i, k=5, n_necessary=1, position="center",
                              necessary_quality=quality, style="record", domain=dom)
                if mode:
                    kwargs["attr_label_mode"] = mode
                items[clabel].append(make_instance(**kwargs))
    return items


def score(clabel, inst, answer):
    a = (answer or "").strip()
    nec = next(r for r in inst.results if r.is_necessary)
    if clabel == "good":
        if fact_in_text(nec.fact, a):
            return "used"
        return "abstain" if (not a or detect_abstention(a)) else "miss"
    if not a or detect_abstention(a):
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


def _load(mid, mock):
    if mock:
        from uptake_probe.models import MockModel
        return MockModel()
    from uptake_probe.models import MLXModel
    return MLXModel(mid, max_tokens=512)


def main():
    args = sys.argv[1:]
    mock = "--mock" in args
    fresh = "--fresh" in args
    n = int(args[args.index("--n") + 1]) if "--n" in args else 15
    items = gen_items(n)
    total = sum(len(v) for v in items.values())
    print(f"precision check: {total} episodes x 2 precisions", flush=True)

    # pick the first pair whose higher-precision checkpoint loads
    runs = []  # (precision_label, model_id)
    if mock:
        runs = [("mock-4bit", "mock"), ("mock-hp", "mock")]
        pair_label = "MockModel"
    else:
        from uptake_probe.models import MLXModel
        pair_label = None
        for plabel, mid4, hp_candidates in PAIRS:
            for hp in hp_candidates:
                try:
                    print(f"probing {hp} ...", flush=True)
                    be = MLXModel(hp, max_tokens=8)
                    be.generate("ping")
                    del be
                    gc.collect()
                    runs = [(f"{plabel}-4bit", mid4), (f"{plabel}-hp", hp)]
                    pair_label = plabel
                    print(f"selected pair: {mid4} vs {hp}", flush=True)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  unavailable: {str(e)[:100]}", flush=True)
            if runs:
                break
        if not runs:
            print("NO-GO: no higher-precision checkpoint loadable"); return

    RAW.parent.mkdir(parents=True, exist_ok=True)
    resume = RAW.exists() and not fresh and not mock
    done = ({(r["precision"], r["cell"], r["instance_id"]) for r in _iter_raw()}
            if resume else set())
    with RAW.open("a" if resume else "w") as fh:
        for plabel, mid in runs:
            todo = [(c, inst) for c, insts in items.items() for inst in insts
                    if (plabel, c, inst.instance_id) not in done]
            print(f"\n### {plabel}: {len(todo)} to run", flush=True)
            if not todo:
                continue
            try:
                be = _load(mid, mock)
            except Exception as e:  # noqa: BLE001
                print(f"  LOAD FAIL: {str(e)[:100]}"); continue
            for clabel, inst in todo:
                ans = run(inst, be, DeliveryMode.AGENTIC, max_steps=20).answer
                fh.write(json.dumps({
                    "precision": plabel, "cell": clabel,
                    "instance_id": inst.instance_id,
                    "outcome": score(clabel, inst, ans),
                    "code_present": has_value_code((ans or "").strip()),
                    "answer": ans}, ensure_ascii=False) + "\n")
                fh.flush()
            del be
            gc.collect()

    _report()


def _report():
    cell = defaultdict(lambda: defaultdict(int))
    for r in _iter_raw():
        cell[(r["precision"], r["cell"])][r["outcome"]] += 1
    precisions = sorted({p for p, _ in cell})
    rows, deltas = [], []
    print(f"\n=== per-cell rates ({' vs '.join(precisions)}) ===")
    for clabel, _, _ in CELLS:
        bad = "used" if clabel == "good" else "confab"
        line = f"  {clabel:<20}"
        vals = {}
        for p in precisions:
            d = cell[(p, clabel)]
            nn = sum(d.values())
            rate = d.get(bad, 0) / nn if nn else 0.0
            lo, hi = wilson_ci(d.get(bad, 0), nn)
            vals[p] = rate
            rows.append({"precision": p, "cell": clabel, "n": nn,
                         "rate": round(rate, 3),
                         "wilson": [round(lo, 3), round(hi, 3)]})
            line += f"  {rate:.2f}"
        if len(vals) == 2:
            a, b = (vals[p] for p in precisions)
            deltas.append((clabel, abs(a - b)))
            line += f"   |d|={abs(a - b):.2f}"
        print(line)
    if deltas:
        mx = max(deltas, key=lambda t: t[1])
        print(f"\nmax |delta| = {mx[1]:.2f} ({mx[0]})")
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"saved -> {OUT} + {RAW}")


if __name__ == "__main__":
    main()
