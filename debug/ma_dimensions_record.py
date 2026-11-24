"""Multi-agent verification across dimensions, RECORD version (replaces the
single-field-era ma_dimensions.py numbers in the paper's §5.6).

Same two-phase design as ma_dimensions.py (each model loaded once, never M*M),
but on the record-style instrument with the core-6 roster, 4 business domains,
and full raw logging (worker episodes AND supervisor judgments) with resume.

  1) workers : each core-6 worker answers n instances per condition
               {irrelevant, attr_mismatch, stale} (record style, 4 domains).
  2) verify  : each core-6 supervisor reads every worker answer + evidence and
               votes GROUNDED/UNSUPPORTED. Catch = flagged real mis-groundings;
               false-alarm = flagged clean answers. Cross-model pairs are the
               headline; self pairs are recorded too.

  uv run --extra mlx python debug/ma_dimensions_record.py            # full
  uv run python debug/ma_dimensions_record.py --smoke                # MockModel
"""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import has_value_code
from uptake_probe.domains import BUSINESS_DOMAINS
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.multiagent import _VERIFY
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance
from uptake_probe.tools import ToolEnv

sys.path.insert(0, str(Path(__file__).parent))
from baseline import ROSTER  # noqa: E402

CORE6 = ["Phi-4-mini", "aya-expanse-8b", "gemma-3-12b",
         "gemma-4-26b", "Qwen3.5-27B", "Qwen3.6-35B"]
CONDS = [ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH, ResultQuality.STALE]
K, N_PER_DOMAIN, BASE = 5, 5, 820000  # 5 x 4 domains = 20 per (worker, cond)
WORK = Path("results/raw/ma_workers_record.jsonl")
SUP = Path("results/raw/ma_record_supervision.jsonl")
OUT = Path("results/ma_dimensions_record.json")


def _roster(smoke):
    if smoke:
        return [("mock", "MockModel")]
    return [(mid, lab) for mid, lab, _ in ROSTER if lab in CORE6]


def _load_backend(mid, smoke):
    if smoke:
        from uptake_probe.models import MockModel
        return MockModel()
    from uptake_probe.models import MLXModel
    return MLXModel(mid, max_tokens=512)


def _instances(cond):
    """Deterministic instance list for one condition: 5 per business domain."""
    out = []
    cell = CONDS.index(cond)
    for di, dom in enumerate(BUSINESS_DOMAINS):
        base = BASE + (cell * len(BUSINESS_DOMAINS) + di) * 1000
        for i in range(N_PER_DOMAIN):
            out.append((base + i, dom,
                        make_instance(seed=base + i, k=K, n_necessary=1,
                                      position="center", necessary_quality=cond,
                                      style="record", domain=dom)))
    return out


def _iter_jsonl(path):
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def phase_workers(models, smoke, fresh):
    WORK.parent.mkdir(parents=True, exist_ok=True)
    resume = WORK.exists() and not fresh and not smoke
    done = ({(r["worker"], r["cond"], r["seed"]) for r in _iter_jsonl(WORK)}
            if resume else set())
    with WORK.open("a" if resume else "w") as fh:
        for mid, label in models:
            todo = [(cond, seed, dom, inst)
                    for cond in CONDS
                    for seed, dom, inst in _instances(cond)
                    if (label, cond.value, seed) not in done]
            print(f"\n### worker {label}: {len(todo)} to run", flush=True)
            if not todo:
                continue
            try:
                be = _load_backend(mid, smoke)
            except Exception as e:  # noqa: BLE001
                print(f"  LOAD FAIL: {str(e)[:90]} -- skip"); continue
            nconf = defaultdict(int)
            for cond, seed, dom, inst in todo:
                rec = run(inst, be, DeliveryMode.AGENTIC, max_steps=20)
                confab = classify_error_awareness(inst, rec) == "confabulate"
                nconf[cond.value] += confab
                fh.write(json.dumps({
                    "worker": label, "cond": cond.value, "domain": dom, "seed": seed,
                    "question": inst.question, "answer": rec.answer,
                    "confab": confab,
                    "code_present": has_value_code((rec.answer or "").strip()),
                    "evidence": ToolEnv(inst).passive_block(),
                }, ensure_ascii=False) + "\n")
                fh.flush()
            for cond in CONDS:
                nn = sum(1 for c, *_ in todo if c == cond)
                if nn:
                    print(f"  {cond.value:<14} worker confab={nconf[cond.value] / nn:.2f}",
                          flush=True)
            del be
            gc.collect()


def phase_verify(models, smoke, fresh):
    workers = list(_iter_jsonl(WORK))
    resume = SUP.exists() and not fresh and not smoke
    done = ({(r["supervisor"], r["worker"], r["cond"], r["seed"])
             for r in _iter_jsonl(SUP)} if resume else set())
    with SUP.open("a" if resume else "w") as fh:
        for mid, sup in models:
            todo = [w for w in workers
                    if (sup, w["worker"], w["cond"], w["seed"]) not in done]
            print(f"\n### supervisor {sup}: {len(todo)} judgments", flush=True)
            if not todo:
                continue
            try:
                be = _load_backend(mid, smoke)
            except Exception as e:  # noqa: BLE001
                print(f"  LOAD FAIL: {str(e)[:90]} -- skip"); continue
            for w in todo:
                prompt = _VERIFY.format(q=w["question"], ans=w["answer"],
                                        evidence=w["evidence"])
                out = be.generate(prompt)
                flagged = "unsupported" in out.lower().split("verdict:", 1)[-1][:40]
                fh.write(json.dumps({
                    "supervisor": sup, "worker": w["worker"], "cond": w["cond"],
                    "seed": w["seed"], "confab": w["confab"], "flagged": flagged,
                }, ensure_ascii=False) + "\n")
                fh.flush()
            del be
            gc.collect()


def report():
    catch = defaultdict(lambda: [0, 0])  # (worker,sup,cond) -> [n_confab, caught]
    fa = defaultdict(lambda: [0, 0])
    for r in _iter_jsonl(SUP):
        key = (r["worker"], r["supervisor"], r["cond"])
        if r["confab"]:
            catch[key][0] += 1
            catch[key][1] += int(r["flagged"])
        else:
            fa[key][0] += 1
            fa[key][1] += int(r["flagged"])
    rows = []
    for (worker, sup, cond) in sorted(set(catch) | set(fa)):
        nc, caught = catch[(worker, sup, cond)]
        ncl, flagged = fa[(worker, sup, cond)]
        rows.append({"worker": worker, "supervisor": sup, "cond": cond,
                     "n_confab": nc, "catch_rate": round(caught / nc, 3) if nc else None,
                     "n_clean": ncl, "false_alarm": round(flagged / ncl, 3) if ncl else None})
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {OUT} ({len(rows)} rows)")
    print("\n=== cross-model catch by condition (record version) ===")
    for cond in (c.value for c in CONDS):
        cross = [(key, v) for key, v in catch.items()
                 if key[2] == cond and key[0] != key[1] and v[0] > 0]
        if cross:
            mean = sum(v[1] / v[0] for _, v in cross) / len(cross)
            k = sum(v[1] for _, v in cross)
            nn = sum(v[0] for _, v in cross)
            fas = [v[1] / v[0] for key, v in fa.items()
                   if key[2] == cond and key[0] != key[1] and v[0] > 0]
            famean = sum(fas) / len(fas) if fas else 0.0
            print(f"  {cond:<14} pair-mean {mean:.2f}  weighted {k}/{nn}={k / nn:.2f}"
                  f"  (cells={len(cross)}, FA {famean:.2f})")


def main() -> None:
    smoke = "--smoke" in sys.argv
    fresh = "--fresh" in sys.argv
    models = _roster(smoke)
    phase_workers(models, smoke, fresh)
    phase_verify(models, smoke, fresh)
    report()


if __name__ == "__main__":
    main()
