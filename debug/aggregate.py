"""Aggregate all capacity_sweep*.json into one cross-model table (no MLX).

Dedupes per (model, mode), preferring the fixed-harness source (capacity_sweep_dl
for aya, which was re-run after the multi-id-fetch fix). Prints the consolidated
agentic-vs-passive table sorted by size, and a crude orthogonality view.

    uv run python debug/aggregate.py
"""

from __future__ import annotations

import json
from pathlib import Path

# later files take precedence on duplicate (model, mode) — dl has the fixed aya
FILES = [
    "capacity_sweep.json",        # clean 5-model (max_tokens=1536)
    "capacity_sweep_4b.json",     # Qwen3.5-4B, Qwen3-4B-2507
    "capacity_sweep_local2.json", # 7 local untested
    "capacity_sweep_dl.json",     # DL batch + fixed aya
]


def main() -> None:
    cells: dict[tuple[str, str], dict] = {}
    for f in FILES:
        p = Path("results") / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            cells[(r["model"], r["mode"])] = r  # later file wins

    # one row per model with both modes
    models: dict[str, dict] = {}
    for (model, mode), r in cells.items():
        m = models.setdefault(model, {"size_b": r["size_b"], "family": r["family"]})
        m[mode] = r

    rows = sorted(models.items(), key=lambda kv: (kv[1]["size_b"], kv[0]))
    print(f"CROSS-MODEL  (N=8, K=12, center)   {len(rows)} models")
    print(f"{'model':<17}{'B':>5} {'fam':<10}"
          f"{'AG.cov':>7}{'AG.upt':>7}{'omit':>6}{'negl':>6}{'steps':>6}  {'PA.cov':>7}")
    print("-" * 76)
    for model, m in rows:
        ag = m.get("agentic", {})
        pa = m.get("passive", {})
        ond = ag.get("ond_share", {})
        print(f"{model:<17}{m['size_b']:>5} {m['family']:<10}"
              f"{ag.get('coverage', float('nan')):>7.3f}{ag.get('uptake_rate', float('nan')):>7.3f}"
              f"{ond.get('omission', float('nan')):>6.2f}{ond.get('neglect', float('nan')):>6.2f}"
              f"{ag.get('avg_steps', float('nan')):>6.1f}  {pa.get('coverage', float('nan')):>7.3f}")

    # crude orthogonality view: agentic coverage by size bucket
    print("\nagentic coverage by size bucket (orthogonality check):")
    buckets: dict[str, list] = {}
    for model, m in rows:
        b = m["size_b"]
        tag = "3-4B" if b <= 4 else "8-9B" if b <= 9 else "12-14B" if b <= 14 else "24-27B"
        buckets.setdefault(tag, []).append((model, m.get("agentic", {}).get("coverage", float("nan"))))
    for tag in ("3-4B", "8-9B", "12-14B", "24-27B"):
        vals = buckets.get(tag, [])
        covs = [c for _, c in vals]
        rng = f"{min(covs):.2f}–{max(covs):.2f}" if covs else "-"
        print(f"  {tag:<7} n={len(vals):<2} agentic-cover range {rng:<12} "
              f"{[(mm, round(c, 2)) for mm, c in vals]}")


if __name__ == "__main__":
    main()
