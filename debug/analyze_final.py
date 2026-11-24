"""Final analysis of the consistent 20-model run (no MLX).

Prints the publishable table, classifies each model's agentic status, and tests
the orthogonality claim: Spearman correlation of agentic coverage vs size, with a
permutation p-value (pure-python). Reports both the full set and the <24B subset
(where the task discriminates).

    uv run python debug/analyze_final.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(0)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j < len(xs) and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            r[order[k]] = avg
        i = j
    return r


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    return cov / (va * vb) if va * vb else 0.0


def spearman(x, y):
    return _pearson(_ranks(x), _ranks(y))


def perm_p(x, y, rho, iters=20000):
    yy = list(y)
    hits = 0
    for _ in range(iters):
        random.shuffle(yy)
        if abs(spearman(x, yy)) >= abs(rho) - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1)


def classify(ag):
    cov = ag.get("coverage", 0.0)
    steps = ag.get("avg_steps", 0.0)
    ond = ag.get("ond_share", {})
    if steps >= 24:
        return "cant-engage"
    if cov >= 0.9:
        return "success"
    if ond.get("neglect", 0) >= ond.get("omission", 0) and ond.get("neglect", 0) > 0.1:
        return "neglect"
    return "omission/partial"


def main() -> None:
    data = json.loads(Path("results/capacity_sweep_final.json").read_text())
    models: dict[str, dict] = {}
    for r in data["rows"]:
        models.setdefault(r["model"], {"size_b": r["size_b"], "family": r["family"]})[r["mode"]] = r
    rows = sorted(models.items(), key=lambda kv: (kv[1]["size_b"], kv[0]))

    print(f"FINAL CROSS-MODEL TABLE  ({len(rows)} models, N=8 K=12 center n=15, fixed harness)\n")
    print(f"{'model':<17}{'B':>5} {'fam':<10}{'AG.cov':>7}{'AG.upt':>7}"
          f"{'omit':>6}{'negl':>6}{'steps':>6}  {'PA.cov':>7}  {'status':<16}")
    print("-" * 90)
    for model, m in rows:
        ag, pa = m.get("agentic", {}), m.get("passive", {})
        ond = ag.get("ond_share", {})
        print(f"{model:<17}{m['size_b']:>5} {m['family']:<10}"
              f"{ag.get('coverage', 0):>7.3f}{ag.get('uptake_rate', 0):>7.3f}"
              f"{ond.get('omission', 0):>6.2f}{ond.get('neglect', 0):>6.2f}{ag.get('avg_steps', 0):>6.1f}"
              f"  {pa.get('coverage', 0):>7.3f}  {classify(ag):<16}")

    # orthogonality: agentic coverage vs size
    sizes = [m["size_b"] for _, m in rows]
    covs = [m.get("agentic", {}).get("coverage", 0.0) for _, m in rows]
    rho = spearman(sizes, covs)
    p = perm_p(sizes, covs, rho)
    print(f"\nSpearman(agentic coverage, size)  ALL n={len(rows)}:  rho={rho:+.3f}  p={p:.3f}")

    sub = [(model, m) for model, m in rows if m["size_b"] < 24]
    s2 = [m["size_b"] for _, m in sub]
    c2 = [m.get("agentic", {}).get("coverage", 0.0) for _, m in sub]
    rho2 = spearman(s2, c2)
    p2 = perm_p(s2, c2, rho2)
    print(f"Spearman(agentic coverage, size)  <24B n={len(sub)}: rho={rho2:+.3f}  p={p2:.3f}")

    # within-bucket range + passive contrast
    print("\nwithin-size-bucket agentic coverage range (orthogonality):")
    buckets: dict[str, list] = {}
    for model, m in rows:
        b = m["size_b"]
        tag = "3-4B" if b <= 4 else "8-9B" if b <= 9 else "12-14B" if b <= 14 else "24-27B"
        buckets.setdefault(tag, []).append(m.get("agentic", {}).get("coverage", 0.0))
    for tag in ("3-4B", "8-9B", "12-14B", "24-27B"):
        v = buckets.get(tag, [])
        if v:
            print(f"  {tag:<7} n={len(v):<2} range {min(v):.2f}-{max(v):.2f}")
    pa_min = min(m.get("passive", {}).get("coverage", 0.0) for _, m in rows)
    print(f"\npassive coverage: min={pa_min:.3f} (comprehension near-universal)")
    statuses = [classify(m.get("agentic", {})) for _, m in rows]
    from collections import Counter
    print("agentic status counts:", dict(Counter(statuses)))


if __name__ == "__main__":
    main()
