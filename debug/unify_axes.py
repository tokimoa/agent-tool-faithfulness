"""Unify the two axes: uptake (does it use good results) vs error-awareness
(does it confabulate on a silent bad result). Joins the consistent uptake run
(capacity_sweep_final.json) with the error-awareness run (error_awareness.json)
and tests whether the two axes — and size — predict each other.

    uv run python debug/unify_axes.py
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


def main() -> None:
    # axis 1: uptake (agentic coverage at N=8)
    up = {}
    for r in json.loads(Path("results/capacity_sweep_final.json").read_text())["rows"]:
        if r["mode"] == "agentic":
            up[r["model"]] = r["coverage"]

    # axis 2: error-awareness (IRRELEVANT confabulation + GOOD control)
    ea = {}
    for r in json.loads(Path("results/error_awareness.json").read_text()):
        d = ea.setdefault(r["model"], {"size": r["size_b"]})
        if r["cond"] == "irrelevant":
            d["confab"] = r["dist"].get("confabulate", 0.0)
        if r["cond"] == "good":
            d["good_ok"] = r["dist"].get("correct", 0.0)

    rows = []
    for model, d in ea.items():
        if "confab" in d:
            rows.append((model, d["size"], up.get(model, float("nan")),
                         d["confab"], d.get("good_ok", 0.0)))
    rows.sort(key=lambda x: x[1])

    print("TWO-AXIS  (axis1 uptake=agentic coverage N=8 | axis2 confab=IRRELEVANT)\n")
    print(f"{'model':<18}{'B':>5}{'uptake':>8}{'confab':>8}{'GOOD-ok':>9}  note")
    print("-" * 62)
    for model, size, upt, conf, gok in rows:
        note = "" if gok >= 0.9 else "(low control — caveat)"
        print(f"{model:<18}{size:>5}{upt:>8.2f}{conf:>8.2f}{gok:>9.2f}  {note}")

    # orthogonality tests on the clean subset (control GOOD-ok >= 0.9)
    clean = [(m, s, u, c) for m, s, u, c, g in rows if g >= 0.9 and u == u]
    sizes = [s for _, s, _, _ in clean]
    upts = [u for _, _, u, _ in clean]
    confs = [c for _, _, _, c in clean]
    print(f"\nclean subset (GOOD-ok>=0.9, n={len(clean)}):")
    r1 = spearman(sizes, confs)
    r2 = spearman(upts, confs)
    print(f"  Spearman(confab, size)   = {r1:+.3f}  p={perm_p(sizes, confs, r1):.3f}")
    print(f"  Spearman(confab, uptake) = {r2:+.3f}  p={perm_p(upts, confs, r2):.3f}")
    print(f"  confab range = {min(confs):.2f}–{max(confs):.2f}")
    print("\n=> if both correlations are weak/non-sig, error-awareness is an "
          "axis orthogonal to BOTH size and uptake skill.")


if __name__ == "__main__":
    main()
