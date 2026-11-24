"""Verify the copy-not-fabricate claim of the paper's mechanism section.

Scoring calls an answer a mis-grounding whenever it asserts ANY string matching
the nonce pattern [A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2} without abstaining -- it
never checks whether that string was on the page. So "is the reported code
present in the returned results?" is an independent question, and this script
answers it by regenerating the label-ablation instances from their seeds
(deterministic) and intersecting each answer's codes with the instance's.

A fabricated code would collide with one of the ~25 codes on the page with
probability ~25 / (26^2 * 10^2)^2 ~ 5e-9, so a 100% on-page rate is evidence of
copying rather than generation.

    uv run python debug/check_copy_source.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uptake_probe.domains import BUSINESS_DOMAINS  # noqa: E402
from uptake_probe.schema import ResultQuality  # noqa: E402
from uptake_probe.synth import make_instance  # noqa: E402

CODE = re.compile(r"\b[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}\b")
MODES = ["wrong", "neutral", "bare", "warn"]
SEED_BASE = 720000          # must match debug/mechanism_record.py
N_PER_CELL = 60
RAW = Path("results/raw/mechanism_record_raw.jsonl")


def on_page_codes() -> dict[str, set[str]]:
    pool, cell = {}, 0
    for mode in MODES:
        for dom in BUSINESS_DOMAINS:
            base = SEED_BASE + cell * 1000
            cell += 1
            for i in range(N_PER_CELL):
                inst = make_instance(seed=base + i, k=5, n_necessary=1, position="center",
                                     necessary_quality=ResultQuality.ATTR_MISMATCH,
                                     style="record", domain=dom, attr_label_mode=mode)
                pool[inst.instance_id] = set(
                    CODE.findall(" ".join(r.full_text for r in inst.results)))
    return pool


def main() -> int:
    pool = on_page_codes()
    rows = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
    total = on_page = novel = unknown = 0
    for r in rows:
        if r["outcome"] != "confab":
            continue
        codes = set(CODE.findall(r["answer"]))
        page = pool.get(r["instance_id"])
        if page is None:
            unknown += 1
            continue
        total += 1
        if codes & page:
            on_page += 1
        else:
            novel += 1
            print(f"  NOVEL {r['model']} {r['instance_id']}: {sorted(codes)}")
    mean = sum(len(v) for v in pool.values()) / len(pool)
    print(f"instances regenerated : {len(pool)} (mean {mean:.1f} codes on the page)")
    print(f"mis-grounded answers  : {total}   (unmatched ids: {unknown})")
    print(f"  code IS on the page : {on_page}/{total} = {on_page / total:.1%}")
    print(f"  code NOT on the page: {novel}/{total} = {novel / total:.1%}")
    return 0 if novel == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
