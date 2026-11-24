"""WORKER x SUPERVISOR catch matrix — isolate self-anchoring from verify ability.

Phase 1: each worker generates answers on IRRELEVANT; keep the cases where it
         confabulated (reported a value absent from the sources).
Phase 2: each supervisor verifies EVERY worker's confabulations against the raw
         evidence. catch[worker][supervisor] = fraction flagged UNSUPPORTED.

Diagonal = self-verification; off-diagonal = cross. If the diagonal is
systematically LOWER than the column's off-diagonal entries, models are blind
specifically to their OWN fabrications (self-anchoring), not bad at verifying in
general. Also reports each supervisor's false-alarm on GOOD (worker-correct).

    uv run --extra mlx python debug/ma2_matrix.py
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

from uptake_probe.detect import detect_abstention, fact_in_text, has_value_code
from uptake_probe.models import MLXModel
from uptake_probe.multiagent import _VERIFY
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance
from uptake_probe.tools import ToolEnv

# confabulator workers (over-trust models)
WORKERS = ["gemma-3-12b", "aya-expanse-8b", "Phi-4-mini", "Ministral-8B"]
SUPERVISORS = WORKERS + ["Qwen3.5-9B"]   # + a reliable cross reference
IDS = {
    "gemma-3-12b": "mlx-community/gemma-3-12b-it-qat-4bit",
    "aya-expanse-8b": "mlx-community/aya-expanse-8b-4bit",
    "Phi-4-mini": "mlx-community/Phi-4-mini-instruct-4bit",
    "Ministral-8B": "mlx-community/Ministral-8B-Instruct-2410-4bit",
    "Qwen3.5-9B": "mlx-community/Qwen3.5-9B-MLX-4bit",
}
K, NINST = 5, 20


def _flag(text: str) -> bool:
    return "unsupported" in text.lower().split("verdict:", 1)[-1][:40]


def main() -> None:
    # Phase 1: collect each worker's confabulations + its correct GOOD answers.
    confabs: dict[str, list] = {}   # worker -> [(instance, answer)]
    goods: dict[str, list] = {}
    for w in WORKERS:
        print(f"[gen] {w}", flush=True)
        m = MLXModel(IDS[w], max_tokens=512)
        cc, gg = [], []
        for i in range(NINST):
            inst = make_instance(seed=6000 + i, k=K, n_necessary=1, position="center",
                                 necessary_quality=ResultQuality.IRRELEVANT)
            a = run(inst, m, DeliveryMode.AGENTIC).answer
            if has_value_code(a) and not detect_abstention(a):
                cc.append((inst, a))
            g = make_instance(seed=6000 + i, k=K, n_necessary=1, position="center",
                              necessary_quality=ResultQuality.GOOD)
            ga = run(g, m, DeliveryMode.AGENTIC).answer
            if fact_in_text(g.necessary_facts[0], ga):
                gg.append((g, ga))
        confabs[w], goods[w] = cc, gg
        print(f"   {w}: {len(cc)} confabs, {len(gg)} good", flush=True)
        del m
        gc.collect()

    # Phase 2: each supervisor verifies every worker's confabulations.
    catch = {s: {} for s in SUPERVISORS}
    fa = {}
    for s in SUPERVISORS:
        print(f"[verify] supervisor {s}", flush=True)
        sup = MLXModel(IDS[s], max_tokens=512)
        for w in WORKERS:
            cc = confabs[w]
            flagged = sum(_flag(sup.generate(_VERIFY.format(q=i.question, ans=a,
                          evidence=ToolEnv(i).passive_block()))) for i, a in cc)
            catch[s][w] = round(flagged / len(cc), 3) if cc else None
        # false alarm on this supervisor's OWN good answers (if it's also a worker)
        gg = goods.get(s, [])
        if gg:
            f = sum(_flag(sup.generate(_VERIFY.format(q=i.question, ans=a,
                    evidence=ToolEnv(i).passive_block()))) for i, a in gg)
            fa[s] = round(f / len(gg), 3)
        del sup
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path("results/ma2_matrix.json").write_text(json.dumps(
        {"catch": catch, "false_alarm": fa,
         "n_confab": {w: len(confabs[w]) for w in WORKERS}}, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 72}\nCATCH MATRIX  rows=supervisor, cols=worker  (diagonal*=self)")
    print(f"{'supervisor':<16}" + "".join(f"{w[:10]:>12}" for w in WORKERS))
    print("-" * 72)
    for s in SUPERVISORS:
        cells = []
        for w in WORKERS:
            v = catch[s][w]
            mark = "*" if s == w else " "
            cells.append(f"{('%.2f' % v) if v is not None else '-':>11}{mark}")
        print(f"{s:<16}" + "".join(cells))
    # self vs cross summary
    selfs = [catch[w][w] for w in WORKERS if catch[w][w] is not None]
    crosses = [catch[s][w] for s in SUPERVISORS for w in WORKERS
               if s != w and catch[s][w] is not None]
    print(f"\nmean catch  SELF(diagonal)={sum(selfs)/len(selfs):.2f}  "
          f"CROSS(off-diagonal)={sum(crosses)/len(crosses):.2f}")
    print("=> SELF << CROSS means self-anchoring (blind to OWN fabrications, not others').")
    print("saved -> results/ma2_matrix.json")


if __name__ == "__main__":
    main()
