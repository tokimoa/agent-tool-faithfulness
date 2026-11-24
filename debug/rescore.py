"""Re-score the saved per-instance raw with the CURRENT classifier, without
re-running any model. This is why suite.py saves raw: when the detector is
improved (e.g. after human validation finds a phrasing gap), the final numbers
update offline. Recomputes confab rate + Wilson/bootstrap CI + decoy/novel source.

    uv run python debug/rescore.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, detect_value, has_value_code
from uptake_probe.stats import bootstrap_ci, wilson_ci

RAW = Path("results/raw/taxonomy_raw.jsonl")
OUT = Path("results/taxonomy_hardened.json")


def _confab(answer: str) -> bool:
    a = answer.strip()
    if not a or detect_abstention(a):
        return False
    return has_value_code(a)


def main() -> None:
    rows = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
    cells = defaultdict(list)
    changed = 0
    for r in rows:
        c = _confab(r["answer"])
        if c != bool(r["auto_confab"]):
            changed += 1
        src = None
        if c:
            src = "decoy" if (r.get("decoy") and detect_value(r["decoy"], r["answer"])) else "novel"
        cells[(r["model"], r["cond"])].append((c, src))

    # preserve size_b from any existing hardened file (raw has no size)
    size = {}
    if OUT.exists():
        for r in json.loads(OUT.read_text()):
            size[r["model"]] = r["size_b"]

    out = []
    for (model, cond), items in cells.items():
        bits = [c for c, _ in items]
        n, k = len(bits), sum(bits)
        wl, wh = wilson_ci(k, n)
        bl, bh = bootstrap_ci(bits)
        src = Counter(s for c, s in items if c)
        out.append({"model": model, "size_b": size.get(model), "cond": cond, "n": n,
                    "confab_rate": round(k / n, 4), "n_confab": k,
                    "wilson_lo": round(wl, 4), "wilson_hi": round(wh, 4),
                    "boot_lo": round(bl, 4), "boot_hi": round(bh, 4),
                    "decoy": src.get("decoy", 0), "novel": src.get("novel", 0)})
    out.sort(key=lambda r: (r["size_b"] if r["size_b"] is not None else 0, r["model"], r["cond"]))
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"re-scored {len(rows)} raw rows; {changed} labels changed vs stored auto_confab")
    print(f"saved -> {OUT}")
    # quick table
    by = defaultdict(dict)
    for r in out:
        by[r["model"]][r["cond"]] = r["confab_rate"]
    order = ["irrelevant", "attr_mismatch", "stale", "partial"]
    print(f"\n{'model':<18}{'irrel':>7}{'attr':>7}{'stale':>7}{'partial':>8}")
    for m in sorted(by, key=lambda m: size.get(m, 0)):
        d = by[m]
        print(f"{m:<18}" + "".join(f"{d.get(c, 0):>7.2f}" for c in order[:3])
              + f"{d.get('partial', 0):>8.2f}")


if __name__ == "__main__":
    main()
