"""Generalise the self-verification blind spot: do MULTIPLE confabulator workers
each fail to catch their OWN confabulations, while a fixed cross-model catches them?

For each worker: generate answers on IRRELEVANT (confab) + GOOD (correct), then
have (a) the worker itself and (b) a fixed cross-model (Qwen3.5-9B) verify each
answer against the raw evidence. Report self vs cross catch on confabulations and
false-alarm on correct answers.

    uv run --extra mlx python debug/ma2_selfblind.py
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

WORKERS = [
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", 12.0),
    ("mlx-community/aya-expanse-8b-4bit", "aya-expanse-8b", 8.0),
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini", 3.8),
]
CROSS = ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B")
K, NINST = 5, 15


def _flag(text: str) -> bool:
    return "unsupported" in text.lower().split("verdict:", 1)[-1][:40]


def _verify(sup, inst, ans):
    ev = ToolEnv(inst).passive_block()
    return _flag(sup.generate(_VERIFY.format(q=inst.question, ans=ans, evidence=ev)))


def main() -> None:
    cross = MLXModel(CROSS[0], max_tokens=512)
    rows = []
    for wmid, wlabel, wsize in WORKERS:
        print(f"\n### worker = {wlabel} ({wsize}B)", flush=True)
        worker = MLXModel(wmid, max_tokens=512)
        confab_cases, good_cases = [], []
        for i in range(NINST):
            inst = make_instance(seed=5000 + i, k=K, n_necessary=1, position="center",
                                 necessary_quality=ResultQuality.IRRELEVANT)
            wa = run(inst, worker, DeliveryMode.AGENTIC).answer
            if has_value_code(wa) and not detect_abstention(wa):
                confab_cases.append((inst, wa))
            g = make_instance(seed=5000 + i, k=K, n_necessary=1, position="center",
                              necessary_quality=ResultQuality.GOOD)
            ga = run(g, worker, DeliveryMode.AGENTIC).answer
            if fact_in_text(g.necessary_facts[0], ga):
                good_cases.append((g, ga))

        cs = sum(_verify(worker, i, a) for i, a in confab_cases)
        cx = sum(_verify(cross, i, a) for i, a in confab_cases)
        fa_s = sum(_verify(worker, i, a) for i, a in good_cases)
        fa_x = sum(_verify(cross, i, a) for i, a in good_cases)
        nc, ng = len(confab_cases), len(good_cases)
        row = {"worker": wlabel, "size_b": wsize, "n_confab": nc, "n_good": ng,
               "catch_self": round(cs / nc, 3) if nc else None,
               "catch_cross": round(cx / nc, 3) if nc else None,
               "fa_self": round(fa_s / ng, 3) if ng else None,
               "fa_cross": round(fa_x / ng, 3) if ng else None}
        rows.append(row)
        print(f"  confab={nc} good={ng} | catch self={row['catch_self']} "
              f"cross={row['catch_cross']} | FA self={row['fa_self']} cross={row['fa_cross']}",
              flush=True)
        del worker
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path("results/ma2_selfblind.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n{'=' * 66}\nSELF vs CROSS verification ({CROSS[1]} as cross)")
    print(f"{'worker':<16}{'B':>5}{'catch_self':>11}{'catch_cross':>12}{'FA_self':>9}{'FA_cross':>9}")
    print("-" * 66)
    for r in rows:
        print(f"{r['worker']:<16}{r['size_b']:>5}{str(r['catch_self']):>11}"
              f"{str(r['catch_cross']):>12}{str(r['fa_self']):>9}{str(r['fa_cross']):>9}")
    print("\n=> self-catch ~0 across workers + cross-catch high = general self-blindness.")


if __name__ == "__main__":
    main()
