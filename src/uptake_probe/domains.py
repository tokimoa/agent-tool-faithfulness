"""Business domains for the benchmark: concrete tool-output flavours that keep the
value a NONCE code (so judge-free exact-match still holds) while dressing the
entity, attribute, and document prose as a real tool result (an order-management
export, a deploy log, a billing record, a helpdesk ticket).

Crucial invariant: every attribute here is one whose real-world value is an opaque
code (tracking IDs, transaction refs, API keys, case refs) — never a guessable
quantity (a balance, a status) — so a model cannot answer from priors and the
planted nonce remains the only ground truth.

``style`` still controls surface form (template = the rigid carrier sentence;
realistic = the domain's natural genres); ``domain`` is an orthogonal axis, so a
template-vs-realistic comparison stays clean WITHIN a domain. The default "access"
domain delegates to the original nonce.py / realistic.py code paths unchanged, so
existing experiments remain reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .nonce import (
    different_attribute,
    make_nonce_fact,
    near_miss_entity,
    nonce_entity,
    nonce_value,
)
from .realistic import (
    realistic_partial_passage,
    realistic_passage,
    realistic_stale_passage,
)
from .schema import NonceFact


class _AccessDomain:
    """The original abstract registry/access domain (systems, clearance codes).

    Delegates to the existing primitives so default behaviour is byte-identical."""

    name = "access"

    def new_fact(self, rng): return make_nonce_fact(rng)
    def entity(self, rng): return nonce_entity(rng)
    def near_miss(self, rng, entity): return near_miss_entity(rng, entity)
    def diff_attr(self, rng, avoid): return different_attribute(rng, avoid)
    def realistic_good(self, rng, e, a, v): return realistic_passage(rng, e, a, v)
    def realistic_stale(self, rng, e, a, old): return realistic_stale_passage(rng, e, a, old)
    def realistic_partial(self, rng, e, a, p): return realistic_partial_passage(rng, e, a, p)
    # the abstract control has no multi-field record form: record == realistic (single field)
    def record_good(self, rng, e, a, v): return realistic_passage(rng, e, a, v)
    def record_attr_mismatch(self, rng, e, asked, da, dv): return realistic_passage(rng, e, da, dv)
    def record_irrelevant(self, rng, oe, a, dv): return realistic_passage(rng, oe, a, dv)
    def record_stale(self, rng, e, a, old): return realistic_stale_passage(rng, e, a, old)
    def record_partial(self, rng, e, a, p): return realistic_partial_passage(rng, e, a, p)


# Code-free metadata lines used to pad a multi-field record with realistic, NON-code
# context (so a `record`-style document reads like a real system export without
# introducing extra value-codes that scoring would have to reason about).
_RECORD_CONTEXT = (
    "status: active", "handler: operations desk", "source: system of record",
    "review: no exceptions noted", "channel: primary feed",
)

# Staleness flags kept in a SEPARATE note line — NOT glued to the value field — so a
# model that copies the bare value field silently mis-grounds (recency confab), while
# one that reads the note abstains. Gluing the marker onto the value (earlier form)
# made every copy carry the caveat and suppressed the recency signal (caught by
# debug/elicit_check.py). Each phrasing is verified to trip detect.detect_abstention,
# so the record still passes structural validation; the bare value field alone does
# NOT trip it. ``{attr}`` names the asked field so the cue is fair to a careful agent.
_STALE_NOTES = (
    "note: the {attr} shown above was superseded after re-issue; the current value "
    "is not included in this export.",
    "flag: the {attr} on file is no longer current (rotated in a later cycle); the "
    "live value is not included here.",
    "the {attr} listed above is no longer valid following a re-issue.",
    "caution: the {attr} here is a superseded entry; the up-to-date value is not part "
    "of this record.",
    "record note: the {attr} above was superseded in a later revision and is no "
    "longer current.",
    "the {attr} shown was superseded; the current {attr} is not included here.",
)

# Frozen, LLM-authored, structurally-VALIDATED frame pool (debug/gen_text_pool.py)
# that diversifies record-style prose ~10x while keeping deterministic fill: the LLM
# only wrote the TEMPLATES (with literal {entity}/{old}/{prefix} placeholders); the
# nonce value/entity/attribute are still injected here, so judge-free scoring and
# seed-reproducibility are untouched. Falls back to the hand-written frames when the
# pool file is absent (keeps tests / fresh checkouts reproducible).
_POOL_PATH = Path(__file__).with_name("text_pool.json")
_POOL: dict | None = None


def _pool(cat: str, name: str):
    global _POOL
    if _POOL is None:
        try:
            _POOL = json.loads(_POOL_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            _POOL = {}
    frames = _POOL.get(cat, {}).get(name)
    return frames or None


@dataclass(frozen=True)
class Domain:
    """A business domain: code-valued attributes + natural document genres.

    Entities read as "{kind} {PREFIX}-{n}" (e.g. "shipment SHP-4417"); a near-miss
    keeps the kind+prefix and changes the number (a maximally plausible mismatch).
    Genres are .format()-ed with {entity},{attribute},{value}/{old}/{prefix},{num}.

    The ``record_*`` methods render a richer MULTI-FIELD document (style="record"):
    a header naming the entity, then several "  - {attribute}: {value}" lines. The
    asked field is one of many, so an agent must SELECT the right field rather than
    copy the only code present — this is where attribute-grounding gets tested under
    realistic tool output. Judge-free invariants are preserved: the gold value still
    appears only where it should, every code stays a nonce, and a partial record
    carries NO complete code (negative control).
    """

    name: str
    kinds: tuple[tuple[str, str], ...]   # (kind word, id prefix)
    attributes: tuple[str, ...]
    good_genres: tuple[str, ...]
    stale_genres: tuple[str, ...]
    partial_genres: tuple[str, ...]
    record_headers: tuple[str, ...] = ()

    def _num(self, rng): return rng.randint(10000, 99999)

    # -- multi-field "record" rendering -------------------------------------
    def _record_header(self, rng, entity: str) -> str:
        tmpls = (_pool("headers", self.name) or list(self.record_headers)
                 or [f"[{self.name} export] {{entity}} — record"])
        return rng.choice(tmpls).format(entity=entity, num=self._num(rng))

    def _context(self, rng, n: int) -> list[str]:
        pool = list(_pool("notes", self.name) or _RECORD_CONTEXT)
        rng.shuffle(pool)
        return pool[:n]

    def _freetext(self, rng) -> list[str]:
        pool = _pool("freetext", self.name)
        return [rng.choice(pool)] if pool else []

    def _siblings(self, rng, exclude: set[str], n: int = 4):
        # use as many of the domain's other code-attributes as available (<=4): a
        # realistic record lists several fields, so the asked one is genuinely "one
        # of many" — the attribute-grounding test under a fuller document.
        pool = [a for a in self.attributes if a not in exclude]
        rng.shuffle(pool)
        return [(a, nonce_value(rng)) for a in pool[:n]]

    def _render_record(self, rng, entity, fields, context) -> str:
        lines = [self._record_header(rng, entity)]
        # a field with attr=None is rendered as a bare value line (P2 "bare" mode)
        lines += [f"  - {val}" if attr is None else f"  - {attr}: {val}"
                  for attr, val in fields]
        lines += [f"  - {c}" for c in context]
        return "\n".join(lines)

    def record_good(self, rng, e, a, v) -> str:
        fields = [(a, v)] + self._siblings(rng, {a})
        rng.shuffle(fields)
        return self._render_record(rng, e, fields, self._context(rng, 2) + self._freetext(rng))

    def record_attr_mismatch(self, rng, e, asked_attr, decoy_attr, decoy_val,
                             label_mode="wrong") -> str:
        # RIGHT entity, but the asked field is ABSENT; the decoy is a sibling field.
        # ``label_mode`` is the P2 counterfactual knob — it varies HOW the decoy value
        # is labelled, to test whether the model actually READS the field label:
        #   "wrong"   : labelled with a real, different attribute (default/shipped)
        #   "neutral" : a generic, non-identifying label ("record value")
        #   "bare"    : no field label at all (a free-standing value line)
        #   "warn"    : an explicit "different field from the one requested" label
        # entity-anchored-copy predicts confab is ~FLAT across these (label not used).
        # All modes keep the asked attribute ABSENT (so it stays a true attr_mismatch).
        if label_mode == "wrong":
            decoy_field = (decoy_attr, decoy_val)
            others = self._siblings(rng, {asked_attr, decoy_attr})
        elif label_mode == "neutral":
            decoy_field = ("record value", decoy_val)
            others = self._siblings(rng, {asked_attr})
        elif label_mode == "bare":
            decoy_field = (None, decoy_val)   # rendered without a label
            others = self._siblings(rng, {asked_attr})
        elif label_mode == "warn":
            decoy_field = ("a different field from the one requested", decoy_val)
            others = self._siblings(rng, {asked_attr})
        else:
            raise ValueError(f"unknown label_mode: {label_mode!r}")
        fields = [decoy_field] + others
        rng.shuffle(fields)
        return self._render_record(rng, e, fields, self._context(rng, 2) + self._freetext(rng))

    def record_irrelevant(self, rng, other_e, shown_attr, decoy_val) -> str:
        # a record about a DIFFERENT entity; the header is the cue.
        fields = [(shown_attr, decoy_val)] + self._siblings(rng, {shown_attr})
        rng.shuffle(fields)
        return self._render_record(rng, other_e, fields, self._context(rng, 2) + self._freetext(rng))

    def record_stale(self, rng, e, a, old) -> str:
        # asked field shows the OLD value as a PLAIN field; the staleness marker is a
        # SEPARATE note (see _STALE_NOTES) so copying the bare value is a silent
        # recency mis-grounding, while reading the note leads to abstention.
        fields = [(a, old)] + self._siblings(rng, {a})
        rng.shuffle(fields)
        note = rng.choice(_STALE_NOTES).format(attr=a)
        ctx = self._context(rng, 1) + [note] + self._freetext(rng)
        rng.shuffle(ctx)  # don't always pin the note to the same position
        return self._render_record(rng, e, fields, ctx)

    def record_partial(self, rng, e, a, prefix) -> str:
        # negative control: NO complete code anywhere — asked field truncated, the
        # rest are code-free metadata lines + a free-text remark (kept code-free so a
        # complete code in the answer can only be a fabrication).
        frames = _pool("partial", self.name)
        trunc = (rng.choice(frames).format(prefix=prefix) if frames else
                 f"{prefix}-???? (truncated; full value withheld from this feed)")
        return self._render_record(rng, e, [(a, trunc)], self._context(rng, 3) + self._freetext(rng))

    def entity(self, rng) -> str:
        kind, prefix = rng.choice(self.kinds)
        return f"{kind} {prefix}-{rng.randint(1000, 9999)}"

    def near_miss(self, rng, entity: str) -> str:
        kind, ident = entity.split(" ", 1)
        prefix = ident.split("-", 1)[0]
        return f"{kind} {prefix}-{rng.randint(1000, 9999)}"

    def diff_attr(self, rng, avoid: str) -> str:
        return rng.choice([a for a in self.attributes if a != avoid])

    def new_fact(self, rng) -> NonceFact:
        return NonceFact(self.entity(rng), rng.choice(self.attributes), nonce_value(rng))

    def realistic_good(self, rng, e, a, v) -> str:
        return rng.choice(self.good_genres).format(entity=e, attribute=a, value=v,
                                                   num=self._num(rng))

    def realistic_stale(self, rng, e, a, old) -> str:
        return rng.choice(self.stale_genres).format(entity=e, attribute=a, old=old,
                                                    num=self._num(rng))

    def realistic_partial(self, rng, e, a, prefix) -> str:
        return rng.choice(self.partial_genres).format(entity=e, attribute=a, prefix=prefix,
                                                      num=self._num(rng))


_LOGISTICS = Domain(
    name="logistics",
    kinds=(("shipment", "SHP"), ("order", "ORD"), ("consignment", "CNS")),
    attributes=("carrier tracking ID", "booking reference", "container ID",
                "customs clearance code", "freight invoice number"),
    good_genres=(
        "[OMS export] {entity} — fulfillment record. The {attribute} is {value} "
        "(synced this cycle). Downstream legs are listed in the lane manifest.",
        "Shipment desk ticket #{num}: ops confirmed the {attribute} for {entity} as "
        "{value}. Marked resolved; reopen if the carrier portal disagrees.",
        "Logistics audit — {entity}. The {attribute} was recorded as {value} against "
        "the booking; no exceptions raised.",
    ),
    stale_genres=(
        "[OMS export] {entity} — fulfillment record. The {attribute} on file reads "
        "{old}. A later note flags it as superseded after re-booking; the current "
        "value is not included in this export.",
        "Shipment desk ticket #{num}: the {attribute} for {entity} was {old}. That leg "
        "has since been rebooked and the value is no longer current; see the live "
        "portal.",
    ),
    partial_genres=(
        "[OMS export] {entity} :: {attribute} = {prefix}-… (truncated). The remaining "
        "characters were clipped by the carrier feed; pull the full record from the "
        "portal.",
        "Shipment desk note: the {attribute} for {entity} begins {prefix}-???? in our "
        "copy; the rest was withheld from the feed.",
    ),
    record_headers=(
        "[OMS export] {entity} — fulfillment record",
        "Logistics system snapshot — {entity} (lane manifest #{num})",
    ),
)

_PAYMENTS = Domain(
    name="payments",
    kinds=(("invoice", "INV"), ("account", "ACCT"), ("settlement", "STL")),
    attributes=("transaction reference", "settlement ID", "authorization code",
                "ledger posting code", "remittance reference"),
    good_genres=(
        "Billing system export — {entity}. The {attribute} is {value}, confirmed by "
        "the clearing batch and retained per policy.",
        "Finance ticket #{num}: reconciliation confirmed the {attribute} for {entity} "
        "as {value}. Posted to the ledger; closed.",
        "Audit trail — {entity}. The {attribute} was logged as {value} with no "
        "discrepancy against the settlement file.",
    ),
    stale_genres=(
        "Billing system export — {entity}. The {attribute} on record was {old}. A "
        "later entry marks it superseded after a re-issue; the current value is not "
        "included here.",
        "Finance ticket #{num}: the {attribute} for {entity} was {old}, but that "
        "posting has since been reversed and the live value is no longer current.",
    ),
    partial_genres=(
        "Billing system export — {entity}. The {attribute} begins {prefix}-???? ; the "
        "trailing block was redacted from this statement and must be pulled from the "
        "clearing system.",
        "Finance note: {entity} :: {attribute} = {prefix}-… (truncated in the export).",
    ),
    record_headers=(
        "Billing system export — {entity}",
        "Ledger record — {entity} (batch #{num})",
    ),
)

_DEVOPS = Domain(
    name="devops",
    kinds=(("service", "SVC"), ("deployment", "DPL"), ("pipeline", "PIP")),
    attributes=("API key", "signing key", "deploy hash", "instance ID",
                "access token"),
    good_genres=(
        "[deploy log] {entity}. The {attribute} on record is {value}, verified at "
        "release time and tracked in the secrets manager index.",
        "Incident ticket #{num}: on-call confirmed the {attribute} for {entity} as "
        "{value}. Mitigation applied; ticket closed.",
        "Infra audit — {entity}. The {attribute} was {value} at the time of review; no "
        "drift detected.",
    ),
    stale_genres=(
        "[deploy log] {entity}. The {attribute} on record was {old}. Rotated during "
        "the last release window; the current value is held in the secrets manager and "
        "not included in this export.",
        "Incident ticket #{num}: the {attribute} for {entity} was {old}, now "
        "superseded after rotation; the live value is no longer current here.",
    ),
    partial_genres=(
        "[deploy log] {entity} :: {attribute} = {prefix}-… (truncated). The remaining "
        "characters were masked in the log; retrieve the full value from the vault.",
        "Infra note: the {attribute} for {entity} begins {prefix}-???? ; the rest was "
        "redacted from this export.",
    ),
    record_headers=(
        "[deploy log] {entity} — service record",
        "Infra inventory snapshot — {entity} (rev #{num})",
    ),
)

_SUPPORT = Domain(
    name="support",
    kinds=(("case", "CASE"), ("ticket", "TKT"), ("account", "CRM")),
    attributes=("case reference", "SLA token", "escalation code",
                "account reference", "contact verification code"),
    good_genres=(
        "Helpdesk export — {entity}. The {attribute} is {value}, set when the record "
        "was opened and confirmed by the duty agent.",
        "CRM note #{num}: agent confirmed the {attribute} for {entity} as {value}. "
        "Synced to the case timeline.",
        "QA audit — {entity}. The {attribute} was recorded as {value}; no handling "
        "exceptions noted.",
    ),
    stale_genres=(
        "Helpdesk export — {entity}. The {attribute} on file was {old}. A later note "
        "flags it superseded after the case was re-opened; the current value is not "
        "included in this export.",
        "CRM note #{num}: the {attribute} for {entity} was {old}, now superseded after "
        "a merge; the live value is no longer current here.",
    ),
    partial_genres=(
        "Helpdesk export — {entity} :: {attribute} = {prefix}-… (truncated). The rest "
        "was withheld from the export; pull the full record from the case system.",
        "CRM note: the {attribute} for {entity} begins {prefix}-???? ; trailing "
        "characters redacted.",
    ),
    record_headers=(
        "Helpdesk export — {entity}",
        "CRM record snapshot — {entity} (case timeline #{num})",
    ),
)

_BUSINESS = {d.name: d for d in (_LOGISTICS, _PAYMENTS, _DEVOPS, _SUPPORT)}
_ACCESS = _AccessDomain()
DOMAINS = ["access", *sorted(_BUSINESS)]
BUSINESS_DOMAINS = sorted(_BUSINESS)


def get_domain(name: str):
    if name == "access":
        return _ACCESS
    if name not in _BUSINESS:
        raise ValueError(f"unknown domain: {name!r} (have {DOMAINS})")
    return _BUSINESS[name]
