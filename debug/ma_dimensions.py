"""Multi-agent verification across the taxonomy DIMENSIONS (extends axis 3).

Now that attribute-mismatch is a near-universal blind spot, the question is:
does adding a VERIFIER agent catch attribute / recency mis-grounding, or do
verifiers share the blind spot (i.e. is the failure fixable by more agents)?

Two phases (so each model is loaded once, never M*M):
  1) workers : each worker answers n instances per condition {irrelevant, attr,
               stale}; store answer + confab bit + the raw evidence.
  2) verify  : each supervisor reads every worker answer + evidence and votes
               GROUNDED/UNSUPPORTED. Compute catch (caught real confabs) and
               false-alarm, per (worker, supervisor, condition).

  uv run --extra mlx python debug/ma_dimensions.py            # full
  uv run python debug/ma_dimensions.py --smoke               # MockModel
"""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.metrics import classify_error_awareness
from uptake_probe.multiagent import _VERIFY
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance
from uptake_probe.tools import ToolEnv

MODELS = [
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini"),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B"),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B"),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b"),
    ("mlx-community/Qwen3.5-27B-4bit", "Qwen3.5-27B"),
]
CONDS = [ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH, ResultQuality.STALE]
K, N, BASE = 5, 20, 9500
WORK = Path("results/raw/ma_workers.jsonl")


def _load_backend(mid, smoke):
    if smoke:
        from uptake_probe.models import MockModel
        return MockModel()
    from uptake_probe.models import MLXModel
    return MLXModel(mid, max_tokens=512)


def phase_workers(models, smoke):
    WORK.parent.mkdir(parents=True, exist_ok=True)
    with WORK.open("w") as fh:
        for mid, label in models:
            print(f"\n### worker {label}", flush=True)
            try:
                be = _load_backend(mid, smoke)
            except Exception as e:  # noqa: BLE001
                print(f"  LOAD FAIL: {str(e)[:90]} -- skip"); continue
            for cond in CONDS:
                nconf = 0
                for i in range(N):
                    inst = make_instance(seed=BASE + i, k=K, n_necessary=1, position="center",
                                         necessary_quality=cond)
                    rec = run(inst, be, DeliveryMode.AGENTIC, max_steps=20)
                    confab = classify_error_awareness(inst, rec) == "confabulate"
                    nconf += confab
                    fh.write(json.dumps({
                        "worker": label, "cond": cond.value, "seed": BASE + i,
                        "question": inst.question, "answer": rec.answer,
                        "confab": confab,
                        "evidence": ToolEnv(inst).passive_block(),
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"  {cond.value:<14} worker confab={nconf / N:.2f}", flush=True)
            del be
            gc.collect()


def phase_verify(models, smoke):
    workers = [json.loads(l) for l in WORK.read_text().splitlines() if l.strip()]
    # catch[(worker,sup,cond)] = [n_confab, n_caught]; fa = [n_clean, n_flagged_clean]
    catch = defaultdict(lambda: [0, 0])
    fa = defaultdict(lambda: [0, 0])
    for mid, sup in models:
        print(f"\n### supervisor {sup}", flush=True)
        try:
            be = _load_backend(mid, smoke)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {str(e)[:90]} -- skip"); continue
        for w in workers:
            prompt = _VERIFY.format(q=w["question"], ans=w["answer"], evidence=w["evidence"])
            out = be.generate(prompt)
            flagged = "unsupported" in out.lower().split("verdict:", 1)[-1][:40]
            key = (w["worker"], sup, w["cond"])
            if w["confab"]:
                catch[key][0] += 1
                catch[key][1] += int(flagged)
            else:
                fa[key][0] += 1
                fa[key][1] += int(flagged)
        del be
        gc.collect()

    rows = []
    keys = set(catch) | set(fa)
    for (worker, sup, cond) in sorted(keys):
        nc, caught = catch[(worker, sup, cond)]
        ncl, flagged = fa[(worker, sup, cond)]
        rows.append({"worker": worker, "supervisor": sup, "cond": cond,
                     "n_confab": nc, "catch_rate": round(caught / nc, 3) if nc else None,
                     "n_clean": ncl, "false_alarm": round(flagged / ncl, 3) if ncl else None})
    Path("results/ma_dimensions.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> results/ma_dimensions.json ({len(rows)} rows)")
    # quick per-condition catch summary (cross-model supervisors only)
    for cond in (c.value for c in CONDS):
        xs = [r["catch_rate"] for r in rows
              if r["cond"] == cond and r["worker"] != r["supervisor"]
              and r["catch_rate"] is not None]
        if xs:
            print(f"  {cond:<14} mean cross-model catch = {sum(xs) / len(xs):.2f} "
                  f"(n_cells={len(xs)})")


def main() -> None:
    smoke = "--smoke" in sys.argv
    models = [("mock", "MockModel")] if smoke else MODELS
    phase_workers(models, smoke)
    phase_verify(models, smoke)


if __name__ == "__main__":
    main()
