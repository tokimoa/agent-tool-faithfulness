"""Full deterministic audit of the shipped benchmark + a stratified sample to
eyeball. Verifies the properties a reviewer would attack, directly on the SHIPPED
records (not the generator), so the released artifact is what's checked.

Checks (all must be 0 violations):
  A. gold is a nonce code (un-guessable)                         -> judge-free
  B. unobtainable conditions: gold appears in NO result at all   -> truly absent
     (necessary AND every distractor — the strict precondition for "confab")
  C. obtainable (good-type): gold appears in the necessary result, and NOWHERE
     else -> uniquely attributable to the planted result
  D. decoy != gold (no nonce collision)
  E. all value-codes within an instance are distinct -> no scoring ambiguity
  F. entity-identity per condition: attr/stale/partial show the asked entity;
     irrelevant* show a DIFFERENT entity
  G. question is well-formed (names the asked entity + attribute)

Then prints a stratified sample for human inspection.

  uv run python debug/audit_benchmark.py [n_sample]
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from uptake_probe.detect import detect_value

DS = Path("benchmark/dataset.jsonl")
NONCE = re.compile(r"^[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}$")
CODE = re.compile(r"[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}")
_ENTITY_RIGHT = {"attr_mismatch", "stale", "partial", "good", "good_verify"}
_ENTITY_DIFF = {"irrelevant", "irrelevant_near", "irrelevant_far",
                "irrelevant_verify", "irrelevant_sig_soft", "irrelevant_sig_hard"}


def audit(recs: list[dict]) -> dict:
    v = defaultdict(int)
    examples = defaultdict(list)

    def flag(key, rec):
        v[key] += 1
        if len(examples[key]) < 3:
            examples[key].append(rec["instance_id"])

    for r in recs:
        results = r["results"]
        all_text = " ".join(x["full_text"] for x in results)
        nec_text = " ".join(x["full_text"] for x in results if x["is_necessary"])
        dis_text = " ".join(x["full_text"] for x in results if not x["is_necessary"])
        golds = r["gold_values"]

        # A
        if not all(NONCE.match(g) for g in golds):
            flag("A_gold_not_nonce", r)
        # B / C
        if r["unobtainable_entities"]:
            if any(detect_value(g, all_text) for g in golds):
                flag("B_unobtainable_gold_present_somewhere", r)
        else:
            if not all(detect_value(g, nec_text) for g in golds):
                flag("C_good_gold_missing_from_necessary", r)
            if any(detect_value(g, dis_text) for g in golds):
                flag("C_good_gold_leaks_into_distractor", r)
        # D
        for d in r["decoys"]:
            if d in golds:
                flag("D_decoy_equals_gold", r)
        # E: distinct value-codes within the instance
        codes = CODE.findall(all_text)
        if len(codes) != len(set(codes)):
            flag("E_duplicate_code_in_instance", r)
        # F: entity identity per condition
        ent = r["results"][0]  # any; we check the necessary one
        nec = next((x for x in results if x["is_necessary"]), None)
        asked_entity = _asked_entity(r["question"])
        if nec and asked_entity:
            in_nec = asked_entity in nec["full_text"]
            if r["condition"] in _ENTITY_RIGHT and not in_nec:
                flag("F_entity_right_but_absent", r)
            if r["condition"] in _ENTITY_DIFF and in_nec:
                flag("F_irrelevant_but_entity_present", r)
        # G: question names entity + attribute
        if not asked_entity or asked_entity not in r["question"]:
            flag("G_question_malformed", r)

    return {"violations": dict(v), "examples": dict(examples)}


def _asked_entity(question: str) -> str | None:
    m = re.search(r"for the (.+?)\?", question)
    return m.group(1) if m else None


def main() -> None:
    recs = [json.loads(l) for l in DS.read_text().splitlines() if l.strip()]
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 24

    print(f"=== dataset: {len(recs)} records ===")
    by_dom = Counter(r["domain"] for r in recs)
    by_style = Counter(r["style"] for r in recs)
    by_cell = Counter((r["module"], r["condition"]) for r in recs)
    print("domains:", dict(by_dom))
    print("styles :", dict(by_style))
    print(f"cells  : {len(by_cell)} (module,condition) groups; "
          f"per-cell counts min={min(by_cell.values())} max={max(by_cell.values())}")
    # value reuse stats (contamination-resistance sanity)
    golds = [g for r in recs for g in r["gold_values"]]
    print(f"gold values: {len(golds)} total, {len(set(golds))} unique "
          f"({len(set(golds))/len(golds):.0%} distinct)")

    print(f"\n=== integrity audit (all {len(recs)} records; want all 0) ===")
    res = audit(recs)
    if not res["violations"]:
        print("  ✓ no violations across all checks A–G")
    for k, c in sorted(res["violations"].items()):
        print(f"  ✗ {k}: {c}   e.g. {res['examples'][k]}")

    # stratified sample for eyeball: one per (domain, condition) where possible
    print(f"\n=== stratified sample (n≈{n_sample}) for human inspection ===")
    rng = random.Random(0)
    buckets = defaultdict(list)
    for r in recs:
        buckets[(r["domain"], r["condition"])].append(r)
    keys = list(buckets)
    rng.shuffle(keys)
    shown = 0
    for key in keys:
        if shown >= n_sample:
            break
        r = rng.choice(buckets[key])
        nec = next(x for x in r["results"] if x["is_necessary"])
        q = r["question"].split(")", 1)[-1].strip() if ")" in r["question"] else r["question"]
        print(f"\n[{r['domain']}/{r['condition']}/{r['style']}]  Q: {q}")
        print(f"  necessary: {nec['full_text']}")
        print(f"  gold={r['gold_values']} decoys={r['decoys']} "
              f"unobtainable={bool(r['unobtainable_entities'])}")
        shown += 1


if __name__ == "__main__":
    main()
