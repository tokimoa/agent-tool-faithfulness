"""MA1: fact-fidelity across a relay pipeline (telephone game), per-hop localised.

Homogeneous pipeline (same model in every role): retriever -> relay -> relay ->
answerer. Plant N facts; measure at which hop each nonce value stops surviving.
Tests whether loss is hop-driven (architecture) and present even for strong models.

    uv run --extra mlx python debug/ma1_fidelity.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

from uptake_probe.models import MLXModel
from uptake_probe.multiagent import pipeline_fidelity, run_pipeline

MODELS = [
    ("mlx-community/Qwen3.5-4B-MLX-4bit", "Qwen3.5-4B", 4.0),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5-9B", 9.0),
    ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", 12.0),
    ("mlx-community/Phi-4-4bit", "Phi-4", 14.0),
    ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-24B", 24.0),
    ("mlx-community/Qwen3.5-27B-4bit", "Qwen3.5-27B", 27.0),
]
# argv[1]=N (more facts = more to lose), argv[2]=compress 0/1 (compression pressure)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
COMPRESS = len(sys.argv) > 2 and sys.argv[2] == "1"
ROLES = ["retriever", "relay", "relay", "relay", "relay", "answerer"]  # 6 hops
K, NINST = N + 4, 15
SUFFIX = f"_n{N}" + ("_compress" if COMPRESS else "")


def main() -> None:
    out = []
    for mid, label, size in MODELS:
        print(f"\n### {label} ({size}B)", flush=True)
        try:
            backend = MLXModel(mid, max_tokens=1024)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL: {e!r}")
            continue
        from uptake_probe.synth import make_instance
        all_hops = []
        for i in range(NINST):
            inst = make_instance(seed=3000 + i, k=K, n_necessary=N, position="center")
            all_hops.append(run_pipeline(inst, ROLES, [backend] * len(ROLES), compress=COMPRESS))
        fid = pipeline_fidelity(all_hops)
        out.append({"model": label, "size_b": size, "fidelity": fid})
        line = "  ".join(f"{c['role'][:5]}{c['hop']}={c['fact_survival']:.2f}" for c in fid)
        print(f"  fact survival by hop: {line}", flush=True)
        del backend
        gc.collect()

    Path("results").mkdir(exist_ok=True)
    Path(f"results/ma1_fidelity{SUFFIX}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}\nFACT SURVIVAL BY HOP (N={N}, {len(ROLES)} hops, "
          f"compress={COMPRESS})")
    print(f"{'model':<18}{'B':>5}" + "".join(f"{'h' + str(i):>7}" for i in range(len(ROLES))))
    print("-" * 60)
    for r in out:
        print(f"{r['model']:<18}{r['size_b']:>5}"
              + "".join(f"{c['fact_survival']:>7.2f}" for c in r["fidelity"]))
    print("\nsaved -> results/ma1_fidelity.json")


if __name__ == "__main__":
    main()
