"""Ecological-validity replication (for Limitations): does the over-trust /
mis-grounding effect survive when the planted mismatch is embedded in natural
document prose (KB article / ticket / inventory / audit memo) instead of the
fixed "The {attribute} for the {entity} is {value}" template?

Same seeds, same nonce values, same classifier — only the carrier surface form
changes (style=template vs realistic), so a paired delta isolates "is the finding
template-bound?". If confab tracks across styles (esp. attr_mismatch staying high
and partial staying ~0), the effect is not a template artifact.

  uv run --extra mlx python debug/realistic_replication.py
  uv run python debug/realistic_replication.py --smoke
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

from uptake_probe.detect import confab_source
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.stats import wilson_ci
from uptake_probe.synth import make_instance

MODELS = [
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini", 3.8),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0),
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", 12.0),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B", 24.0),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b", 26.0),
]
CONDS = [ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH,
         ResultQuality.STALE, ResultQuality.PARTIAL]
STYLES = ["template", "realistic"]
K, N, BASE = 5, 30, 9500
RAW = Path("results/raw/realistic_raw.jsonl")


def main() -> None:
    smoke = "--smoke" in sys.argv
    n = 4 if smoke else N
    roster = [("mock", "MockModel", 0.0)] if smoke else MODELS
    # optional condition filter (e.g. "stale") -> re-run that cell and MERGE into
    # the existing results, leaving the other conditions untouched.
    only = [a for a in sys.argv[1:] if a in {c.value for c in CONDS}]
    conds = [c for c in CONDS if c.value in only] if only else CONDS
    merge = bool(only)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_mode = "a" if merge else "w"
    with RAW.open(raw_mode) as raw_fh:
        for mid, label, size in roster:
            print(f"\n### {label} ({size}B)", flush=True)
            if smoke:
                from uptake_probe.models import MockModel
                backend = MockModel()
            else:
                from uptake_probe.models import MLXModel
                try:
                    backend = MLXModel(mid, max_tokens=512)
                except Exception as e:  # noqa: BLE001
                    print(f"  LOAD FAIL ({type(e).__name__}): {str(e)[:100]} -- skip"); continue
            for cond in conds:
                for style in STYLES:
                    bits, src = [], Counter()
                    for i in range(n):
                        inst = make_instance(seed=BASE + i, k=K, n_necessary=1, position="center",
                                             necessary_quality=cond, style=style)
                        rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
                        confab = classify_error_awareness(inst, rec) == "confabulate"
                        s = confab_source(inst, rec.answer) if confab else None
                        if s:
                            src[s] += 1
                        bits.append(confab)
                        raw_fh.write(json.dumps({
                            "model": label, "cond": cond.value, "style": style,
                            "seed": BASE + i, "answer": rec.answer, "confab": confab, "src": s,
                        }, ensure_ascii=False) + "\n")
                        raw_fh.flush()
                    k = sum(bits)
                    lo, hi = wilson_ci(k, n)
                    rows.append({"model": label, "size_b": size, "cond": cond.value,
                                 "style": style, "n": n, "confab_rate": round(k / n, 3),
                                 "wilson_lo": round(lo, 3), "wilson_hi": round(hi, 3),
                                 "decoy": src.get("decoy", 0), "novel": src.get("novel", 0)})
                    print(f"  {cond.value:<14} {style:<9} confab={k / n:.2f} [{lo:.2f},{hi:.2f}]",
                          flush=True)
            del backend
            gc.collect()

    out_path = Path("results/realistic.json")
    if merge and out_path.exists():
        prior = json.loads(out_path.read_text())
        keys = {(r["model"], r["cond"], r["style"]) for r in rows}
        prior = [r for r in prior if (r["model"], r["cond"], r["style"]) not in keys]
        rows = prior + rows
        rows.sort(key=lambda r: (r["size_b"], r["model"], r["cond"], r["style"]))
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    # paired summary: template vs realistic per (model, cond)
    print(f"\n{'=' * 72}\nTEMPLATE vs REALISTIC (confab rate; |delta| small ⇒ not a template artifact)")
    print(f"{'model':<18}{'cond':<14}{'tmpl':>7}{'real':>7}{'delta':>8}")
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["cond"]), {})[r["style"]] = r["confab_rate"]
    deltas = []
    for (m, c), d in by.items():
        t, rl = d.get("template", 0), d.get("realistic", 0)
        deltas.append(abs(t - rl))
        print(f"{m:<18}{c:<14}{t:>7.2f}{rl:>7.2f}{rl - t:>+8.2f}")
    if deltas:
        print(f"\nmean |template-realistic| = {sum(deltas) / len(deltas):.3f}")
    print("saved -> results/realistic.json")


if __name__ == "__main__":
    main()
