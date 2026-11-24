"""C2: does an explicit 'verify the entity matches' instruction cure silent-mismatch
confabulation? If confab drops sharply, the failure is a VERIFICATION OMISSION
(the agent CAN check but doesn't), not a capability limit — and there is a cheap
mitigation. We also check GOOD to ensure the instruction doesn't over-induce
rejection of valid results (the under-trust side effect).

    uv run --extra mlx python debug/mechanism_c2.py
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

from uptake_probe.detect import fact_in_text
from uptake_probe.metrics import classify_error_awareness
from uptake_probe.models import MLXModel
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

MODELS = [
    ("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4-mini", 3.8),
    ("mlx-community/aya-expanse-8b-4bit", "aya-expanse-8b", 8.0),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0),
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", 12.0),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B", 24.0),
]
K, NINST = 5, 20


def main() -> None:
    rows = []
    for mid, label, size in MODELS:
        print(f"\n### {label} ({size}B)", flush=True)
        try:
            backend = MLXModel(mid, max_tokens=512)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {e!r}")
            continue
        rec_d = {}
        for verify in (False, True):
            confab = good_ok = 0
            for i in range(NINST):
                bad = make_instance(seed=8000 + i, k=K, n_necessary=1, position="center",
                                    necessary_quality=ResultQuality.IRRELEVANT,
                                    verify_entity=verify)
                if classify_error_awareness(bad, run(bad, backend, DeliveryMode.AGENTIC,
                                                     max_steps=20)) == "confabulate":
                    confab += 1
                g = make_instance(seed=8000 + i, k=K, n_necessary=1, position="center",
                                  necessary_quality=ResultQuality.GOOD, verify_entity=verify)
                if fact_in_text(g.necessary_facts[0], run(g, backend, DeliveryMode.AGENTIC).answer):
                    good_ok += 1
            rec_d[verify] = (confab / NINST, good_ok / NINST)
            print(f"  verify={verify!s:<5} IRREL-confab={confab / NINST:.2f}  "
                  f"GOOD-ok={good_ok / NINST:.2f}", flush=True)
        rows.append({"model": label, "size_b": size,
                     "confab_off": rec_d[False][0], "confab_on": rec_d[True][0],
                     "good_off": rec_d[False][1], "good_on": rec_d[True][1]})
        del backend
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path("results/mechanism_c2.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 72}\nC2 entity-verification intervention (confab should drop; GOOD should hold)")
    print(f"{'model':<18}{'B':>5}{'confab off→on':>16}{'GOOD off→on':>16}")
    print("-" * 72)
    for r in rows:
        conf = f"{r['confab_off']:.2f}->{r['confab_on']:.2f}"
        good = f"{r['good_off']:.2f}->{r['good_on']:.2f}"
        print(f"{r['model']:<18}{r['size_b']:>5}{conf:>16}{good:>16}")
    print("\n=> big confab drop with GOOD preserved ⇒ verification-omission mechanism + cheap fix.")
    print("saved -> results/mechanism_c2.json")


if __name__ == "__main__":
    main()
