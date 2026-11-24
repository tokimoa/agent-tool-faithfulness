"""Minimal pilot runner (design doc §10, revised per §14).

Grid: delivery mode x K at a fixed (center) position, N necessary facts, with an
optional contradictory distractor. Reports Uptake / Coverage / OND / contradiction
-following per cell, plus cost (tokens, wall time) to size the full experiment.

Default backend is the MockModel SIMULATOR (CPU, no GPU) so the harness and
metrics are exercised end-to-end. Pass --mlx <model_id> for the real backend.

  uv run python -m uptake_probe.pilot
  uv run python -m uptake_probe.pilot --mlx mlx-community/Qwen3-8B-4bit --instances 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .metrics import evaluate_run, summarize
from .models import Backend, MockModel
from .scaffolds import run
from .schema import DeliveryMode
from .synth import make_instance

_MODES = {
    "passive": DeliveryMode.PASSIVE,
    "agentic": DeliveryMode.AGENTIC,
    "passive_long": DeliveryMode.PASSIVE_LONG,
}


def run_pilot(
    backend: Backend,
    ks: tuple[int, ...] = (1, 4, 8),
    modes: tuple[str, ...] = ("passive", "agentic"),
    n_instances: int = 50,
    n_necessary: int = 1,
    position: str = "center",
    include_contradictory: bool = False,
    authority_cue: bool = True,
    base_seed: int = 1000,
    pad_to_tokens: int = 0,
) -> dict:
    cells: list[dict] = []
    t0 = time.perf_counter()
    for mode_name in modes:
        mode = _MODES[mode_name]
        for k in ks:
            if n_necessary > k:
                continue
            per_instance = []
            tokens = []
            steps = []
            for i in range(n_instances):
                inst = make_instance(
                    seed=base_seed + i, k=k, n_necessary=n_necessary,
                    position=position, include_contradictory=include_contradictory,
                    authority_cue=authority_cue,
                )
                rec = run(inst, backend, mode, pad_to_tokens=pad_to_tokens)
                per_instance.append(evaluate_run(inst, rec))
                tokens.append(rec.total_tokens)
                steps.append(rec.steps)
            summary = summarize(per_instance)
            summary.update({
                "mode": mode_name,
                "k": k,
                "avg_tokens": round(sum(tokens) / len(tokens), 1),
                "avg_steps": round(sum(steps) / len(steps), 2),
            })
            cells.append(summary)
    elapsed = time.perf_counter() - t0
    return {
        "backend": backend.name,
        "config": {
            "ks": list(ks), "modes": list(modes), "n_instances": n_instances,
            "n_necessary": n_necessary, "position": position,
            "include_contradictory": include_contradictory,
            "authority_cue": authority_cue, "base_seed": base_seed,
        },
        "elapsed_sec": round(elapsed, 3),
        "cells": cells,
    }


def _print_table(report: dict) -> None:
    print(f"\nbackend={report['backend']}  elapsed={report['elapsed_sec']}s  "
          f"(n={report['config']['n_instances']}/cell)")
    header = f"{'mode':<13}{'K':>3}  {'uptake':>7}{'cover':>7}  " \
             f"{'omit':>6}{'negl':>6}{'distr':>6}  {'CFR':>6}  {'tok':>7}{'steps':>6}"
    print(header)
    print("-" * len(header))
    for c in report["cells"]:
        ond = c.get("ond_share", {})
        cfr = c.get("contradiction_following_rate")
        print(f"{c['mode']:<13}{c['k']:>3}  "
              f"{c['uptake_rate']:>7.3f}{c['coverage']:>7.3f}  "
              f"{ond.get('omission', 0):>6.3f}{ond.get('neglect', 0):>6.3f}"
              f"{ond.get('distraction', 0):>6.3f}  "
              f"{(cfr if cfr is not None else float('nan')):>6.3f}  "
              f"{c['avg_tokens']:>7.0f}{c['avg_steps']:>6.2f}")


def _build_backend(args) -> Backend:
    if args.mlx:
        from .models import MLXModel
        return MLXModel(args.mlx, max_tokens=args.max_tokens)
    return MockModel(model_seed=args.model_seed)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Tool-Result Uptake pilot")
    ap.add_argument("--mlx", default=None, help="MLX model id (else MockModel simulator)")
    ap.add_argument("--instances", type=int, default=50)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--modes", nargs="+", default=["passive", "agentic"], choices=list(_MODES))
    ap.add_argument("--n-necessary", type=int, default=1)
    ap.add_argument("--position", default="center")
    ap.add_argument("--contradictory", action="store_true")
    ap.add_argument("--no-authority-cue", dest="authority_cue", action="store_false",
                    help="drop the explicit 'use the official registry' rule (harder RQ4)")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1536,
                    help="generous to avoid passive answer truncation (see MLXModel)")
    ap.add_argument("--out", default=None, help="path to save JSON report")
    args = ap.parse_args(argv)

    backend = _build_backend(args)
    report = run_pilot(
        backend, ks=tuple(args.ks), modes=tuple(args.modes),
        n_instances=args.instances, n_necessary=args.n_necessary,
        position=args.position, include_contradictory=args.contradictory,
        authority_cue=args.authority_cue,
    )
    _print_table(report)

    out = args.out or f"results/pilot_{backend.name}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
