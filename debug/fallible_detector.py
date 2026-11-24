"""Fallible mismatch detectors in front of the input ladder (added in the camera-ready revision).

Reviewers pointed out that the *isolate* rung assumes an oracle that
knows which record is mismatched. What happens with a detector of MEASURED
accuracy? This script (1) runs small LLMs as a tool-layer validator on the SAME
ladder instances (one call per record; the validator sees the request and the
returned record only), measures their false-negative rate on the three silent
mismatches and false-positive rate on GOOD, and (2) chains them OFFLINE with the
already-logged agent episodes: for every (agent model, instance) the chained
outcome is the logged *isolate* episode when the detector flags the record and
the logged *orig* episode otherwise. On GOOD a false positive withholds the
value, which we score as lost coverage (the isolate rungs show agents emit ~no
code once the value is absent). No agent episodes are re-run.

  uv run --extra mlx python debug/fallible_detector.py --detectors Phi-4-mini,gemma-4-26b
  uv run python debug/fallible_detector.py --lexical      # deterministic reference detector
  uv run python debug/fallible_detector.py --report       # offline chaining only
"""

from __future__ import annotations

import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.stats import wilson_ci

sys.path.insert(0, str(Path(__file__).parent))
from baseline import ROSTER  # noqa: E402
import input_cure  # noqa: E402
import ladder_dims  # noqa: E402

RAW = Path("results/raw/fallible_detector_raw.jsonl")
OUT = Path("results/fallible_detector.json")
CURE_RAW = Path("results/raw/input_cure_raw.jsonl")
DIMS_RAW = Path("results/raw/ladder_dims_raw.jsonl")

PROMPT = """You are a validation layer between a records database and an assistant. \
The assistant requested the CURRENT value of one field for one entity. Decide whether \
the returned record actually provides that: the record must be about exactly the \
requested entity, must contain exactly the requested field (not a different field), \
and the value must not be marked as superseded, outdated, or no longer current.

Requested entity: {entity}
Requested field: {attribute}

Returned record:
{record}

Reply on the first line with exactly `VERDICT: MATCH` if the record provides a current \
value of the requested field for the requested entity, or `VERDICT: MISMATCH` otherwise. \
Then give one short sentence of justification."""

VERDICT = re.compile(r"VERDICT:\s*(MATCH|MISMATCH)", re.I)


def items():
    """{cond: [(instance, asked_entity, asked_attr, record_text)]} matching the raw logs."""
    out = {}
    attr = input_cure.gen_items(15)
    out["attr_mismatch"] = [(i, ladder_dims._asked_entity(i), input_cure._asked(i),
                             ladder_dims._nec(i).full_text) for i in attr]
    dims = ladder_dims.gen_items(15)
    for cond, insts in dims.items():
        out[cond] = [(i, ladder_dims._asked_entity(i), ladder_dims._nec(i).fact.attribute,
                      ladder_dims._nec(i).full_text) for i in insts]
    return out


class LexicalDetector:
    """Deterministic reference: flag if the asked field label is absent from the record,
    the asked entity is absent from the header, or a supersession note is present."""
    name = "lexical"

    def verdict(self, entity, attribute, record):
        header = record.split("\n")[0]
        has_field = re.search(r"^\s*-\s*" + re.escape(attribute) + r"\s*:", record, re.M | re.I)
        has_entity = entity.lower() in header.lower()
        stale = re.search(r"superseded|no longer current|outdated", record, re.I)
        return "MISMATCH" if (not has_field or not has_entity or stale) else "MATCH"


# Approximate 4-bit weight footprint (GB) used for the pre-load memory guard;
# another MLX server may be resident on the same machine.
# Small non-roster checkpoints usable as detectors (already cached locally).
EXTRA_DETECTORS = {"gemma-4-e4b": "mlx-community/gemma-4-e4b-it-4bit"}
_FOOTPRINT_GB = {"gemma-4-e4b": 4.0, "Phi-4-mini": 2.5, "aya-expanse-8b": 5.0, "gemma-3-12b": 8.0, "Phi-4": 8.5,
                 "Mistral-Small-24B": 14.0, "gemma-4-26b": 15.0, "Qwen3.5-27B": 16.0,
                 "Qwen3.6-35B": 20.0}


def _available_gb():
    """System-wide free memory (GB) as macOS memory_pressure reports it."""
    import subprocess
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                               text=True).stdout) / 1e9
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    pct = int(re.search(r"free percentage:\s*(\d+)%", out).group(1))
    return total * pct / 100


class LLMDetector:
    def __init__(self, mid, label):
        import mlx.core as mx
        from uptake_probe.models import MLXModel
        import os
        need = _FOOTPRINT_GB.get(label, 16.0) * float(os.environ.get("DETECTOR_GUARD", "1.3"))
        avail = _available_gb()
        if avail < need:
            raise MemoryError(f"skip {label}: need ~{need:.0f} GB, available {avail:.1f} GB")
        mx.set_cache_limit(256 * 1024 * 1024)   # keep the Metal cache from ballooning
        self.be = MLXModel(mid, max_tokens=96)
        self.name = label

    def verdict(self, entity, attribute, record):
        txt = self.be.generate(PROMPT.format(entity=entity, attribute=attribute, record=record))
        m = VERDICT.search(txt)
        return (m.group(1).upper() if m else "UNPARSED"), txt


def _iter_raw():
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def run_detectors(labels, lexical=False):
    its = items()
    done = {(r["detector"], r["cond"], r["instance_id"]) for r in _iter_raw()}
    dets = []
    if lexical:
        dets.append(("lexical", None))
    known = {lab: mid for mid, lab, _ in ROSTER}
    known.update(EXTRA_DETECTORS)
    dets += [(lab, known[lab]) for lab in labels if lab in known]
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with RAW.open("a") as fh:
        for label, mid in dets:
            todo = [(c, t) for c, ts in its.items() for t in ts if (label, c, t[0].instance_id) not in done]
            print(f"\n### detector {label}: {len(todo)} records", flush=True)
            if not todo:
                continue
            try:
                det = LexicalDetector() if mid is None else LLMDetector(mid, label)
            except MemoryError as e:
                print(f"  {e} -- not loaded, resume later", flush=True)
                continue
            for k, (cond, (inst, ent, attr, rec)) in enumerate(todo):
                v = det.verdict(ent, attr, rec)
                verdict, raw = (v, "") if isinstance(v, str) else v
                fh.write(json.dumps({"detector": label, "cond": cond,
                                     "instance_id": inst.instance_id, "verdict": verdict,
                                     "raw": raw[:300]}, ensure_ascii=False) + "\n")
                fh.flush()
                if k % 40 == 0:
                    print(f"  {k}/{len(todo)} {cond} -> {verdict}", flush=True)
            del det
            gc.collect()


def report():
    # detector verdicts
    ver = defaultdict(dict)  # detector -> (cond, iid) -> flagged?
    for r in _iter_raw():
        ver[r["detector"]][(r["cond"], r["instance_id"])] = (r["verdict"] == "MISMATCH")
    # logged agent episodes: (model, cond, iid, lever) -> row
    ep = {}
    for line in CURE_RAW.read_text().splitlines():
        r = json.loads(line)
        ep[(r["model"], "attr_mismatch", r["instance_id"], r["lever"])] = r
    for line in DIMS_RAW.read_text().splitlines():
        r = json.loads(line)
        ep[(r["model"], r["cond"], r["instance_id"], r["lever"])] = r
    models = [lab for _, lab, _ in ROSTER if any(k[0] == lab for k in ep)]
    conds = ["irrelevant", "attr_mismatch", "stale"]
    summary = {}
    for det, flags in ver.items():
        print(f"\n===== detector: {det} =====")
        s = {"detector": det, "detector_rates": {}, "chained": {}}
        for cond in conds + ["good"]:
            keys = [k for k in flags if k[0] == cond]
            n = len(keys)
            fl = sum(flags[k] for k in keys)
            if cond == "good":
                rate, name = fl / n, "false_positive"
            else:
                rate, name = 1 - fl / n, "false_negative"
            lo, hi = wilson_ci(fl if cond == "good" else n - fl, n)
            s["detector_rates"][cond] = {name: round(rate, 3), "n": n, "wilson": [round(lo, 3), round(hi, 3)]}
            print(f"  {cond:14} {name} = {rate:.3f} [{lo:.2f},{hi:.2f}]  (n={n})")
        print(f"  {'chained mis-grounding (orig -> chained), code_present':<60}")
        pooled = defaultdict(lambda: [0, 0, 0, 0])  # cond -> [orig_bad, chained_bad, chained_code, n]
        for m in models:
            row = {}
            for cond in conds:
                ob = cb = cc = n = 0
                for (mm, c, iid, lev), r in ep.items():
                    if mm != m or c != cond or lev != "orig":
                        continue
                    if (cond, iid) not in flags:
                        continue
                    n += 1
                    ob += r["outcome"] == "confab"
                    use = ep.get((m, cond, iid, "isolate"), r) if flags[(cond, iid)] else r
                    cb += use["outcome"] == "confab"
                    cc += bool(use.get("code_present", use["outcome"] == "confab"))
                if n:
                    row[cond] = {"orig": round(ob / n, 3), "chained": round(cb / n, 3),
                                 "chained_code_present": round(cc / n, 3), "n": n}
                    p = pooled[cond]; p[0] += ob; p[1] += cb; p[2] += cc; p[3] += n
            # GOOD: chained utilization = used AND not flagged
            ou = cu = n = 0
            for (mm, c, iid, lev), r in ep.items():
                if mm != m or c != "good" or lev != "orig" or ("good", iid) not in flags:
                    continue
                n += 1
                ou += r["outcome"] == "used"
                cu += (r["outcome"] == "used") and not flags[("good", iid)]
            if n:
                row["good"] = {"orig_util": round(ou / n, 3), "chained_util": round(cu / n, 3), "n": n}
                p = pooled["good"]; p[0] += ou; p[1] += cu; p[3] += n
            s["chained"][m] = row
            cells = "  ".join(f"{c[:4]} {row[c]['orig']:.2f}->{row[c]['chained']:.2f}(cp{row[c]['chained_code_present']:.2f})"
                              for c in conds if c in row)
            g = row.get("good", {})
            print(f"  {m:<18} {cells}  good util {g.get('orig_util', 0):.2f}->{g.get('chained_util', 0):.2f}")
        print("  --- pooled ---")
        s["pooled"] = {}
        for cond in conds:
            ob, cb, cc, n = pooled[cond]
            lo, hi = wilson_ci(cb, n)
            s["pooled"][cond] = {"orig": round(ob / n, 3), "chained": round(cb / n, 3),
                                 "wilson": [round(lo, 3), round(hi, 3)],
                                 "chained_code_present": round(cc / n, 3), "n": n}
            print(f"  {cond:14} {ob/n:.2f} -> {cb/n:.2f} [{lo:.2f},{hi:.2f}]  code_present {cc/n:.2f}  (n={n})")
        ou, cu, _, n = pooled["good"]
        s["pooled"]["good"] = {"orig_util": round(ou / n, 3), "chained_util": round(cu / n, 3), "n": n}
        print(f"  {'good util':14} {ou/n:.2f} -> {cu/n:.2f}  (n={n})")
        summary[det] = s
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {OUT}")


def main():
    args = sys.argv[1:]
    if "--report" not in args:
        labels = args[args.index("--detectors") + 1].split(",") if "--detectors" in args else []
        run_detectors(labels, lexical="--lexical" in args)
    report()


if __name__ == "__main__":
    main()
