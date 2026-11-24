"""Generate a DIVERSE, frozen text-frame pool for the record-style documents,
fully LOCAL, with deterministic-fill semantics preserved.

Why: the deterministic generator only had ~3 hand-written prose frames per domain,
so across thousands of records the same sentence templates repeat ~120x (measured).
That reads as mechanical. Here a local LLM AUTHORS many diverse frames (headers,
notes, stale/partial annotations) as TEMPLATES with literal placeholders; we then
STRUCTURALLY VALIDATE each frame and freeze only the clean ones. The nonce value /
entity / attribute are still injected by the deterministic generator at fill time,
so judge-free scoring and "regenerate from seed" are untouched — the LLM never sees
or writes a value.

Pipeline per category x domain:
  generate (temp>0, several rounds) -> parse lines -> validate -> dedup -> accumulate
Output -> src/uptake_probe/text_pool.json  (shipped with the generator)

  uv run --extra mlx python debug/gen_text_pool.py [--model ID] [--per N] [--dry]

``--dry`` runs the validators over a canned sample (no GPU) to self-test the gates.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, has_value_code
from uptake_probe.domains import BUSINESS_DOMAINS, get_domain

OUT = Path("src/uptake_probe/text_pool.json")
DEFAULT_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
_CODE = re.compile(r"[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}")
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_TRUNC_HINT = re.compile(r"\?\?\?\?|…|\btrunc|\bwithheld|\bredacted|\bmasked|\bclipped", re.I)
# the planted value is ALWAYS an opaque alphanumeric code (e.g. VX26-YQ43), never a
# quantity. Reject any frame that frames it as money/number — a prior LLMs leak for
# "payments" (calling a transaction reference an "amount"/"figure").
_BAD_SEMANTICS = re.compile(
    r"\b(amount|amounts|figure|figures|balance|sum|total|totals|price|prices|cost|"
    r"costs|dollar|dollars|currency|quantity|tally)\b|[\$€£]|\bUSD\b|\bEUR\b", re.I)


# --- validators (pure; offline-testable) -------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _clean_line(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", s)   # bullets / numbering
    s = s.strip().strip("`").strip('"').strip()
    return _norm(s)


def _placeholders(s: str) -> list[str]:
    return _PLACEHOLDER.findall(s)


def valid_header(s: str) -> bool:
    if not (8 <= len(s) <= 120) or _CODE.search(s) or _BAD_SEMANTICS.search(s):
        return False
    ph = _placeholders(s)
    return ph.count("entity") == 1 and set(ph) <= {"entity", "num"}


def valid_note(s: str) -> bool:
    return (3 <= len(s) <= 90 and not _CODE.search(s) and not _placeholders(s)
            and not _BAD_SEMANTICS.search(s))


def valid_stale(s: str) -> bool:
    if _CODE.search(s) or _BAD_SEMANTICS.search(s):
        return False
    ph = _placeholders(s)
    if ph.count("old") != 1 or set(ph) != {"old"}:
        return False
    return detect_abstention(s.replace("{old}", "VX26-YQ43"))  # marker must be present


def valid_partial(s: str) -> bool:
    if _CODE.search(s) or _BAD_SEMANTICS.search(s):
        return False
    ph = _placeholders(s)
    if ph.count("prefix") != 1 or set(ph) != {"prefix"}:
        return False
    return bool(_TRUNC_HINT.search(s))


def valid_freetext(s: str) -> bool:
    # a one-sentence operational remark: code-free, placeholder-free, no quantities.
    return (15 <= len(s) <= 120 and not _CODE.search(s) and not _placeholders(s)
            and not _BAD_SEMANTICS.search(s))


_VALIDATORS = {"headers": valid_header, "notes": valid_note,
               "stale": valid_stale, "partial": valid_partial,
               "freetext": valid_freetext}


# --- prompts -----------------------------------------------------------------

def _prompt(category: str, domain, n: int) -> str:
    name = domain.name
    sysd = (f"domain: {name}. example record subjects (entities): "
            f"'{domain.entity.__doc__ or ''}' e.g. a {name} system export/ticket/log.")
    if category == "headers":
        return (
            f"You write headers for {name} system records (an export, ticket, log, "
            f"audit, or snapshot). Output EXACTLY {n} DISTINCT one-line headers, one "
            f"per line, no numbering. Each header MUST contain the literal token "
            f"{{entity}} exactly once where the record's subject goes. You MAY use "
            f"{{num}} once for a record/ticket number. Vary the system name, wording, "
            f"and format. Do NOT include any code or value. Example shape: "
            f"'[OMS export] {{entity}} — fulfillment record'.")
    if category == "notes":
        return (
            f"You write short status/metadata lines that appear inside a {name} "
            f"record. Output EXACTLY {n} DISTINCT short lines, one per line, no "
            f"numbering. Each is a brief field like 'status: in transit', 'reviewed "
            f"by the duty desk', 'channel: primary feed'. NO value codes, NO "
            f"placeholders, under ~8 words.")
    if category == "freetext":
        return (
            f"You write one-sentence operational remarks found at the bottom of a "
            f"{name} record (handling notes, follow-ups, caveats). Output EXACTLY {n} "
            f"DISTINCT natural sentences, one per line, no numbering. Do NOT name any "
            f"specific field or include any code, number, amount, or placeholder. "
            f"Example: 'Handlers should confirm scope before acting on this record.'")
    if category == "stale":
        return (
            f"You write annotations marking a {name} field value as OUTDATED. The value "
            f"is an OPAQUE ALPHANUMERIC CODE (like 'VX26-YQ43') — NEVER money, an "
            f"amount, a figure, or any quantity; call it 'value', 'code', or 'entry'. "
            f"Output EXACTLY {n} DISTINCT annotations, one per line, no numbering. Each "
            f"MUST contain the literal token {{old}} exactly once (the old code) and "
            f"MUST make clear it is superseded / no longer current and the current "
            f"value is NOT included here. Vary the wording. NO value codes besides the "
            f"{{old}} token. Example: '{{old}} — superseded after re-issue; the current "
            f"value is not included in this export'.")
    if category == "partial":
        return (
            f"You write annotations showing a {name} field value that is TRUNCATED. The "
            f"value is an OPAQUE ALPHANUMERIC CODE (like 'VX26-YQ43') — NEVER money, an "
            f"amount, or a quantity; call it 'value', 'code', or 'string'. Output "
            f"EXACTLY {n} DISTINCT annotations, one per line, no numbering. Each MUST "
            f"contain the literal token {{prefix}} exactly once, followed by a "
            f"truncation marker like '????' and wording that the rest was withheld / "
            f"redacted / must be pulled from the source. NO complete value code. "
            f"Example: '{{prefix}}-???? (truncated; full value withheld from this feed)'.")
    raise ValueError(category)


# --- driver ------------------------------------------------------------------

def _canned():
    return {
        "headers": ["1. [OMS export] {entity} — fulfillment record",
                    "- Logistics snapshot for {entity} (ref #{num})",
                    "bad {entity} {entity}", "no placeholder here",
                    "has a code AB12-CD34 for {entity}"],
        "notes": ["status: in transit", "- reviewed by the duty desk",
                  "has {placeholder}", "AB12-CD34 code line"],
        "stale": ["{old} — superseded after re-issue; current value not included here",
                  "{old} is fine and current",                # no marker -> reject
                  "{old} and {extra} placeholder"],
        "partial": ["{prefix}-???? (truncated; rest withheld)",
                    "{prefix} complete AB12-CD34",            # has code -> reject
                    "{prefix} no truncation hint"],
        "freetext": ["Handlers should confirm scope before acting on this record.",
                     "short",                                  # too short -> reject
                     "Outstanding balance of 50 dollars remains."],  # quantity -> reject
    }


def main() -> None:
    args = sys.argv[1:]
    dry = "--dry" in args
    per = int(args[args.index("--per") + 1]) if "--per" in args else 22
    model = args[args.index("--model") + 1] if "--model" in args else DEFAULT_MODEL

    if dry:
        print("=== DRY: self-testing validators on canned LLM-ish output ===")
        for cat, lines in _canned().items():
            v = _VALIDATORS[cat]
            for raw in lines:
                s = _clean_line(raw)
                print(f"  [{cat:8}] valid={str(v(s)):5}  {s!r}")
        return

    from uptake_probe.models import MLXModel
    be = MLXModel(model, max_tokens=900, temperature=0.8)
    pool: dict[str, dict[str, list[str]]] = {c: {} for c in _VALIDATORS}
    for name in BUSINESS_DOMAINS:
        dom = get_domain(name)
        for cat, validate in _VALIDATORS.items():
            seen: set[str] = set()
            kept: list[str] = []
            rounds = 0
            while len(kept) < per and rounds < 6:
                rounds += 1
                out = be.generate(_prompt(cat, dom, per))
                for raw in out.splitlines():
                    s = _clean_line(raw)
                    key = s.lower()
                    if s and key not in seen and validate(s):
                        seen.add(key)
                        kept.append(s)
            pool[cat][name] = kept[:per]
            print(f"  {name:10} {cat:8} kept {len(pool[cat][name]):2}/{per} "
                  f"in {rounds} rounds", flush=True)
    OUT.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    total = sum(len(v) for c in pool.values() for v in c.values())
    print(f"\nsaved -> {OUT}  ({total} validated frames)")


if __name__ == "__main__":
    main()
