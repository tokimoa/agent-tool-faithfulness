"""Error-awareness axis: when a tool result is BAD (the asked value is
unobtainable), does the agent abstain/recover, or confabulate a value and
proceed 'satisfied'?

Conditions on the single necessary result (N=1, K=5, agentic):
  GOOD       -> control: the value IS available (should answer correctly)
  ERROR      -> "record not found"
  EMPTY      -> no content
  IRRELEVANT -> a passage about a DIFFERENT entity (value absent)
  REDACTED   -> value present but [REDACTED]

For bad conditions we report abstain / recover / confabulate / vague.
Key question: do models that ACE the uptake axis also abstain here, or confabulate?

    uv run --extra mlx python debug/error_awareness.py
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

from uptake_probe.detect import fact_in_text
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

MODELS = [  # agent-capable models (Llama family excluded: can't engage ReAct)
    ("mlx-community/Ministral-3-3B-Instruct-2512-4bit", "Ministral-3-3B", 3.0),
    ("mlx-community/Phi-3.5-mini-instruct-4bit", "Phi-3.5-mini", 3.8),
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini", 3.8),
    ("mlx-community/Qwen3.5-4B-MLX-4bit", "Qwen3.5-4B", 4.0),
    ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "Qwen3-4B-2507", 4.0),
    ("mlx-community/gemma-3-4b-it-qat-4bit", "gemma-3-4b", 4.0),
    ("mlx-community/aya-expanse-8b-4bit", "aya-expanse-8b", 8.0),
    ("mlx-community/Qwen3-8B-4bit", "Qwen3-8B", 8.0),
    ("mlx-community/Ministral-8B-Instruct-2410-4bit", "Ministral-8B", 8.0),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0),
    ("mlx-community/GLM-4-9B-0414-4bit", "GLM-4-9B", 9.0),
    ("mlx-community/Yi-1.5-9B-Chat-4bit", "Yi-1.5-9B", 9.0),
    ("mlx-community/Mistral-Nemo-Instruct-2407-4bit", "Mistral-Nemo", 12.0),
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", 12.0),
    ("mlx-community/Phi-4-4bit", "Phi-4", 14.0),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B", 24.0),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b", 26.0),
    ("mlx-community/Qwen3.5-27B-4bit", "Qwen3.5-27B", 27.0),
]
# focused conditions: control + explicit-failure + the discriminating silent mismatch
CONDS = [ResultQuality.GOOD, ResultQuality.ERROR, ResultQuality.IRRELEVANT]
K = 5
# argv[1]: n_instances (smoke with a small number). argv[2]: model labels csv.
NINST = int(sys.argv[1]) if len(sys.argv) > 1 else 20
if len(sys.argv) > 2:
    _want = sys.argv[2].split(",")
    MODELS = [m for m in MODELS if m[1] in _want]
SMOKE = NINST < 10


def main() -> None:
    rows = []
    raw = []  # per-run records, so labels can be re-derived offline if the classifier changes
    for mid, label, size in MODELS:
        print(f"\n### {label} ({size}B)", flush=True)
        try:
            backend = MLXModel(mid, max_tokens=512)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {e!r}")
            continue
        for q in CONDS:
            labels = []
            for i in range(NINST):
                inst = make_instance(seed=2000 + i, k=K, n_necessary=1, position="center",
                                     necessary_quality=q)
                # generous step budget so recovery can conclude (abstain) rather
                # than hit the cap mid-search and return an empty answer.
                rec = run(inst, backend, DeliveryMode.AGENTIC, max_steps=20)
                if q is ResultQuality.GOOD:
                    ok = fact_in_text(inst.necessary_facts[0], rec.answer)
                    lab = "correct" if ok else "miss"
                else:
                    lab = classify_error_awareness(inst, rec)
                labels.append(lab)
                raw.append({"model": label, "size_b": size, "cond": q.value, "seed": 2000 + i,
                            "label": lab, "answer": rec.answer[:400],
                            "searches": rec.searches, "steps": rec.steps})
                if SMOKE:
                    print(f"    [{q.value}] {lab:<11} steps={rec.steps} "
                          f"searches={rec.searches} :: {rec.answer[:140]!r}", flush=True)
            dist = {k: round(v / NINST, 3) for k, v in Counter(labels).items()}
            rows.append({"model": label, "size_b": size, "cond": q.value, "dist": dist})
            print(f"  {q.value:<11} {dist}", flush=True)
        del backend
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path("results/error_awareness.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    Path("results/error_awareness_raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False))

    # summary: confabulation rate under bad conditions (the headline)
    print(f"\n{'=' * 64}\nCONFABULATION RATE under bad results  (IRREL = silent mismatch)")
    print(f"{'model':<18}{'B':>5} {'ERR-conf':>9}{'IRREL-conf':>11}  {'GOOD-ok':>8}")
    by_model: dict = {}
    sizes: dict = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["cond"]] = r["dist"]
        sizes[r["model"]] = r["size_b"]
    for model in by_model:
        d = by_model[model]
        def conf(c):
            return d.get(c, {}).get("confabulate", 0.0)
        print(f"{model:<18}{sizes[model]:>5}"
              f"{conf('error'):>9.2f}{conf('irrelevant'):>11.2f}"
              f"  {d.get('good', {}).get('correct', 0.0):>8.2f}")
    print("\nsaved -> results/error_awareness.json")


if __name__ == "__main__":
    main()
