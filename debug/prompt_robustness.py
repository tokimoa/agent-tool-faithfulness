"""Prompt-robustness: is the mis-grounding result an artifact of our exact system
prompt? Re-run the two headline conditions under 3 differently-worded agentic
system prompts (same SEARCH/FETCH/ANSWER protocol). If confab is stable across
wordings, the finding is not prompt-specific.

  uv run --extra mlx python debug/prompt_robustness.py
  uv run python debug/prompt_robustness.py --smoke
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import uptake_probe.scaffolds as scaffolds
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

# variant A = the original (scaffolds._AGENTIC_HEADER); B and C reword the same protocol.
HEADER_B = (
    "ROLE: you are a tool-using assistant. The answer is NOT given to you; you must "
    "fetch it with tools, one tool call per turn on its own line beginning 'Action:'.\n"
    "TOOLS:\n"
    "  Action: SEARCH[query] -> returns available result ids + snippets\n"
    "  Action: FETCH[id]     -> returns a result's full text (ids look like r0, r1)\n"
    "  Action: ANSWER[text]  -> submit the final answer and stop\n"
    "WORKFLOW: search once, then fetch each result you need, then answer with the "
    "exact values. Do not nest SEARCH/FETCH inside ANSWER. Never invent data.\n"
)
HEADER_C = (
    "You operate a retrieval interface and must ground every answer in fetched text.\n"
    "Emit exactly one command per line, prefixed 'Action:'. Available commands: "
    "SEARCH[query] to list result ids, FETCH[id] to read a result in full, and "
    "ANSWER[text] to finish. Begin by searching, then read the relevant results, "
    "then report the values precisely. If a value is not present in what you fetched, "
    "say so rather than guessing.\n"
)
VARIANTS = {"orig": scaffolds._AGENTIC_HEADER, "reword_B": HEADER_B, "reword_C": HEADER_C}

MODELS = [
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini"),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B"),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B"),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b"),
]
CONDS = [ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH]
K, N, BASE = 5, 20, 9500
_ORIG = scaffolds._AGENTIC_HEADER


def _confab_rate(backend, cond, n):
    c = 0
    for i in range(n):
        inst = make_instance(seed=BASE + i, k=K, n_necessary=1, position="center",
                             necessary_quality=cond)
        if classify_error_awareness(inst, run(inst, backend, DeliveryMode.AGENTIC,
                                              max_steps=20)) == "confabulate":
            c += 1
    return c / n


def main() -> None:
    smoke = "--smoke" in sys.argv
    models = [("mock", "MockModel")] if smoke else MODELS
    rows = []
    try:
        for mid, label in models:
            print(f"\n### {label}", flush=True)
            if smoke:
                from uptake_probe.models import MockModel
                backend = MockModel()
            else:
                from uptake_probe.models import MLXModel
                try:
                    backend = MLXModel(mid, max_tokens=512)
                except Exception as e:  # noqa: BLE001
                    print(f"  LOAD FAIL: {str(e)[:90]} -- skip"); continue
            for vname, header in VARIANTS.items():
                scaffolds._AGENTIC_HEADER = header      # monkeypatch the prompt
                for cond in CONDS:
                    r = _confab_rate(backend, cond, N)
                    rows.append({"model": label, "variant": vname, "cond": cond.value,
                                 "confab_rate": round(r, 3)})
                    print(f"  {vname:<9} {cond.value:<14} confab={r:.2f}", flush=True)
            del backend
            gc.collect()
    finally:
        scaffolds._AGENTIC_HEADER = _ORIG               # always restore

    Path("results/prompt_robustness.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    # per (model,cond): spread across prompt variants (small spread = robust)
    print(f"\n{'model':<18}{'cond':<14}{'min':>6}{'max':>6}{'spread':>8}")
    agg = {}
    for r in rows:
        agg.setdefault((r["model"], r["cond"]), []).append(r["confab_rate"])
    for (m, c), vs in agg.items():
        print(f"{m:<18}{c:<14}{min(vs):>6.2f}{max(vs):>6.2f}{max(vs) - min(vs):>8.2f}")
    print("saved -> results/prompt_robustness.json")


if __name__ == "__main__":
    main()
