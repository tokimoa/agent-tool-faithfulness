"""Project the unified hardened run (suite_full.json, n=50+CI) into the per-figure
JSON schemas the figures already expect, so fig5/6/7/8/9 all become hardened+CI
from ONE coherent dataset (same seeds, same human-validated classifier).

  mechanism.json        <- C1 cells  irrelevant_near / irrelevant / irrelevant_far
  mechanism_c2.json     <- C2: off=irrelevant, on=irrelevant_verify; good off/on
  lexical_signal.json   <- C5: none=irrelevant, soft=lexical_soft, hard=lexical_hard
  taxonomy_hardened.json<- B:  irrelevant / attr_mismatch / stale / partial (+CIs)

Backs up the prior n=20 files to results/_pre_hardened/ first.

  uv run python debug/suite_full_to_figs.py
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

SRC = Path("results/suite_full.json")
BACKUP = Path("results/_pre_hardened")


def main() -> None:
    rows = json.loads(SRC.read_text())
    # index: by (model, cell) -> row
    by = {(r["model"], r["cell"]): r for r in rows}
    size = {r["model"]: r["size_b"] for r in rows}
    models = [m for m in dict.fromkeys(r["model"] for r in rows)]

    BACKUP.mkdir(exist_ok=True)
    for f in ("mechanism.json", "mechanism_c2.json", "lexical_signal.json",
              "taxonomy_hardened.json"):
        p = Path("results") / f
        if p.exists():
            shutil.copy(p, BACKUP / f)

    def rate(m, cell):
        r = by.get((m, cell))
        return r["rate"] if r else None

    def share(m, cell):
        r = by.get((m, cell))
        if not r:
            return None
        tot = r["decoy"] + r["novel"]
        return round(r["decoy"] / tot, 3) if tot else None

    # --- mechanism.json (C1 plausibility gradient) ---
    mech = []
    for m in models:
        for cell, cond in [("irrelevant_near", "irrelevant_near"),
                           ("irrelevant", "irrelevant"),
                           ("irrelevant_far", "irrelevant_far")]:
            r = by.get((m, cell))
            if r:
                mech.append({"model": m, "size_b": size[m], "cond": cond,
                             "confab_rate": r["rate"], "n_confab": r["k"],
                             "decoy_share": share(m, cell)})
    Path("results/mechanism.json").write_text(json.dumps(mech, indent=2, ensure_ascii=False))

    # --- mechanism_c2.json (C2 verify-entity intervention) ---
    c2 = []
    for m in models:
        c2.append({"model": m, "size_b": size[m],
                   "confab_off": rate(m, "irrelevant"), "confab_on": rate(m, "irrelevant_verify"),
                   "good_off": rate(m, "good"), "good_on": rate(m, "good_verify")})
    Path("results/mechanism_c2.json").write_text(json.dumps(c2, indent=2, ensure_ascii=False))

    # --- lexical_signal.json (C5) ---
    lex = []
    for m in models:
        for sig, cell in [("none", "irrelevant"), ("soft", "lexical_soft"),
                          ("hard", "lexical_hard")]:
            r = by.get((m, cell))
            if r:
                lex.append({"model": m, "size_b": size[m], "signal": sig,
                            "confab_rate": r["rate"], "n_confab": r["k"],
                            "decoy_share": share(m, cell)})
    Path("results/lexical_signal.json").write_text(json.dumps(lex, indent=2, ensure_ascii=False))

    # --- taxonomy_hardened.json (B, keep CIs) ---
    tax = []
    for m in models:
        for cell in ("irrelevant", "attr_mismatch", "stale", "partial"):
            r = by.get((m, cell))
            if r:
                tax.append({"model": m, "size_b": size[m], "cond": cell, "n": r["n"],
                            "confab_rate": r["rate"], "n_confab": r["k"],
                            "wilson_lo": r["wilson_lo"], "wilson_hi": r["wilson_hi"],
                            "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"],
                            "decoy": r["decoy"], "novel": r["novel"]})
    Path("results/taxonomy_hardened.json").write_text(json.dumps(tax, indent=2, ensure_ascii=False))

    print(f"projected {len(rows)} cells -> mechanism/mechanism_c2/lexical_signal/"
          f"taxonomy_hardened.json (prior n=20 backed up to {BACKUP}/)")
    print(f"models: {len(models)}")


if __name__ == "__main__":
    main()
