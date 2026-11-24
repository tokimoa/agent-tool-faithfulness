"""Robustness checks that close three reviewer objections to the headline
mis-grounding result. Run on a small representative set spanning the profiles.

  config   : does confab survive other operating points? K in {3,5,8} x position
             in {early,center,late} (defuses "single cherry-picked operating point").
  delivery : agentic ReAct vs PASSIVE (results pre-loaded in context). If passive
             confab is comparable, the failure is NOT a ReAct-scaffold artifact
             (defuses the Llama-excluded interface confound).
  temp     : temp>0, k samples/instance -> mean rate + pass^k consistency (fraction
             of instances whose k samples all agree). Defuses "temp=0 single sample".

  uv run --extra mlx python debug/robustness.py            # all three
  uv run --extra mlx python debug/robustness.py config     # one section
  uv run python debug/robustness.py --smoke                # MockModel, no GPU
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

from uptake_probe.metrics import classify_error_awareness
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

# representative models spanning the dimension profiles (entity/attr/recency)
REPS = [
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini"),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B"),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B"),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b"),
]
HEAD = [ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH]
BASE = 9500


def _confab(inst, backend, mode=DeliveryMode.AGENTIC) -> bool:
    rec = run(inst, backend, mode, max_steps=20)
    return classify_error_awareness(inst, rec) == "confabulate"


def sweep_config(backend, label, n=15):
    out = []
    for cond in HEAD:
        for k in (3, 5, 8):
            for pos in ("early", "center", "late"):
                c = sum(_confab(make_instance(seed=BASE + i, k=k, n_necessary=1, position=pos,
                                              necessary_quality=cond), backend)
                        for i in range(n))
                out.append({"exp": "config", "model": label, "cond": cond.value,
                            "k": k, "position": pos, "n": n, "confab_rate": round(c / n, 3)})
                print(f"  [config] {cond.value:<13} K={k} {pos:<6} confab={c / n:.2f}", flush=True)
    return out


def sweep_delivery(backend, label, n=20):
    out = []
    for cond in HEAD:
        for mode in (DeliveryMode.AGENTIC, DeliveryMode.PASSIVE):
            c = sum(_confab(make_instance(seed=BASE + i, k=5, n_necessary=1, position="center",
                                          necessary_quality=cond), backend, mode)
                    for i in range(n))
            out.append({"exp": "delivery", "model": label, "cond": cond.value,
                        "delivery": mode.value, "n": n, "confab_rate": round(c / n, 3)})
            print(f"  [delivery] {cond.value:<13} {mode.value:<8} confab={c / n:.2f}", flush=True)
    return out


def sweep_temp(backend, label, n=15, ksamp=5):
    """temp>0 stability: mean rate over k samples + pass^k consistency."""
    out = []
    for cond in HEAD:
        rates, consistent = [], 0
        for i in range(n):
            inst = make_instance(seed=BASE + i, k=5, n_necessary=1, position="center",
                                 necessary_quality=cond)
            bits = [_confab(inst, backend) for _ in range(ksamp)]
            rates.append(sum(bits) / ksamp)
            if len(set(bits)) == 1:
                consistent += 1
        out.append({"exp": "temp", "model": label, "cond": cond.value, "n": n,
                    "ksamp": ksamp, "mean_confab": round(sum(rates) / n, 3),
                    "passk_consistency": round(consistent / n, 3)})
        print(f"  [temp] {cond.value:<13} mean={sum(rates) / n:.2f} "
              f"pass^{ksamp}-consistency={consistent / n:.2f}", flush=True)
    return out


def main() -> None:
    smoke = "--smoke" in sys.argv
    which = [a for a in sys.argv[1:] if a in ("config", "delivery", "temp")]
    sections = which or ["config", "delivery", "temp"]

    if smoke:
        from uptake_probe.models import MockModel
        reps = [("mock", "MockModel")]
    else:
        reps = REPS

    rows = []
    for mid, label in reps:
        print(f"\n### {label}", flush=True)
        if smoke:
            backend = MockModel()
        else:
            from uptake_probe.models import MLXModel
            try:
                backend = MLXModel(mid, max_tokens=512)
            except Exception as e:  # noqa: BLE001
                print(f"  LOAD FAIL ({type(e).__name__}): {str(e)[:110]} -- skip", flush=True)
                continue
        if "config" in sections:
            rows += sweep_config(backend, label)
        if "delivery" in sections:
            rows += sweep_delivery(backend, label)
        if "temp" in sections:
            if not smoke:  # reload at temp>0 so config/delivery stay deterministic
                from uptake_probe.models import MLXModel
                del backend
                gc.collect()
                backend = MLXModel(mid, max_tokens=512, temperature=0.7)
            rows += sweep_temp(backend, label)
        del backend
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path("results/robustness.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> results/robustness.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
