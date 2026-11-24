"""Choose a LOCAL QC judge: which on-device model best gates realistic-document
naturalness? We want a model that (a) follows the binary rubric, (b) is
CONSERVATIVE — it must accept genuinely-natural docs (low false-AWKWARD) and only
flag truly broken ones, and (c) does NOT invent content (a known failure of cheap
judges). Everything stays local so generation+QC need no API.

Probe set per model:
  - 12 GOOD realistic docs (all genres)  -> should be ~all NATURAL
  - 12 deliberately BROKEN docs (word-salad / truncated mid-word / duplicated
    fragments)                            -> a good judge flags most as AWKWARD
Report: natural-rate on good (want high), awkward-rate on broken (want high),
and a balanced score. Also time per judgment (throughput for ~thousands of docs).

  uv run --extra mlx python debug/pick_qc_judge.py
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

from uptake_probe.nonce import make_nonce_fact
from uptake_probe.qc import judge_naturalness
from uptake_probe.realistic import realistic_passage

CANDIDATES = [
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B"),
    ("mlx-community/Phi-4-4bit", "Phi-4-14B"),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-24B"),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b"),
    ("mlx-community/Qwen3.5-27B-4bit", "Qwen3.5-27B"),
]


def _good_docs(rng, n=12):
    out = []
    for _ in range(n):
        f = make_nonce_fact(rng)
        out.append(realistic_passage(rng, f.entity, f.attribute, f.value))
    return out


def _broken_docs(rng, n=12):
    out = []
    for _ in range(n):
        f = make_nonce_fact(rng)
        good = realistic_passage(rng, f.entity, f.attribute, f.value)
        kind = rng.choice(["salad", "midword", "dup", "empty"])
        if kind == "salad":
            w = good.split()
            rng.shuffle(w)
            out.append(" ".join(w))
        elif kind == "midword":
            out.append(good[: len(good) // 2] + "xqzve th")
        elif kind == "dup":
            frag = good[:40]
            out.append((frag + " ") * 4)
        else:
            out.append(".")
    return out


def _borderline_docs(rng, n=12):
    """Slightly-off but NOT broken: subtly stilted phrasing a conservative judge
    should still KEEP (these probe over-rejection — the dangerous direction)."""
    tmpl = [
        "Record for {e}. The {a} is {v}. End of record.",
        "{e}: {a} -> {v}. (auto-generated entry, do not edit)",
        "Note. {a} of {e} equals {v}. Status nominal.",
        "FYI the {a} for {e} is {v}. thanks",
    ]
    out = []
    for _ in range(n):
        f = make_nonce_fact(rng)
        out.append(rng.choice(tmpl).format(e=f.entity, a=f.attribute, v=f.value))
    return out


def main() -> None:
    rng = random.Random(0)
    good = _good_docs(rng)
    broken = _broken_docs(rng)
    border = _borderline_docs(rng)   # should be KEPT (probe over-rejection)
    rows = []
    for mid, label in CANDIDATES:
        print(f"\n### {label}", flush=True)
        try:
            from uptake_probe.models import MLXModel
            be = MLXModel(mid, max_tokens=64)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {str(e)[:90]}"); continue
        gn = sum(judge_naturalness(be, d).natural for d in good)
        ba = sum(not judge_naturalness(be, d).natural for d in broken)
        bn = sum(judge_naturalness(be, d).natural for d in border)
        good_rate, broken_rate = gn / len(good), ba / len(broken)
        border_keep = bn / len(border)
        # weight conservatism (keeping good+borderline) since false-AWKWARD biases data
        balanced = round((good_rate + broken_rate + border_keep) / 3, 3)
        rows.append({"model": label, "good_natural_rate": round(good_rate, 3),
                     "broken_awkward_rate": round(broken_rate, 3),
                     "borderline_keep_rate": round(border_keep, 3), "balanced": balanced})
        print(f"  good→NAT {good_rate:.2f}  broken→AWK {broken_rate:.2f}  "
              f"borderline→KEEP {border_keep:.2f}  balanced {balanced:.2f}", flush=True)
        del be
        gc.collect()

    rows.sort(key=lambda r: (r["balanced"], r["borderline_keep_rate"]), reverse=True)
    Path("results/qc_judge_pick.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n{'=' * 60}\nLOCAL QC JUDGE RANKING (balanced = mean of the two rates)")
    for r in rows:
        print(f"  {r['model']:<14} good {r['good_natural_rate']:.2f} | "
              f"broken {r['broken_awkward_rate']:.2f} | balanced {r['balanced']:.2f}")
    if rows:
        print(f"\n=> recommend: {rows[0]['model']} (conservative + discriminating)")
    print("saved -> results/qc_judge_pick.json")


if __name__ == "__main__":
    main()
