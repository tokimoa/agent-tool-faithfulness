"""LLM-as-a-judge NATURALNESS gate for realistic-style documents.

This is the ONLY place the benchmark pipeline uses an LLM, and it is used at
GENERATION time, never at evaluation time, and its decisions are baked into the
released (filtered) dataset — so the shipped benchmark and its scoring stay fully
judge-free. The judge only answers a narrow, low-risk yes/no: "does this read like
a plausible internal business record?" All semantic/structural correctness is
already guaranteed deterministically by validate.py before a doc reaches here.

The backend is pluggable (any object with ``.generate(prompt) -> str``): a local
MLX model for full reproducibility, or a frontier API for a stronger gate. We log
the model id + prompt + pass-rate so the QC itself is documented and auditable
(and a conservative judge that errs toward KEEP avoids the "judge invents data"
failure mode seen with some cheap models).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Backend

_QC_PROMPT = (
    "You are a data-quality reviewer for a benchmark. Below is a single retrieved "
    "document that will be shown to an AI agent. Judge ONLY whether it reads as a "
    "plausible, fluent internal business/IT record (e.g. a knowledge-base entry, "
    "support ticket, inventory export, or audit log). Do NOT judge correctness, "
    "and do NOT rewrite it. Ignore the specific codes/identifiers — they are "
    "deliberately random.\n"
    "Answer on the first line with exactly 'VERDICT: NATURAL' or 'VERDICT: AWKWARD'.\n\n"
    "DOCUMENT:\n{doc}\n\nVERDICT:"
)
_VERDICT = re.compile(r"verdict:\s*(natural|awkward)", re.IGNORECASE)


@dataclass
class QCResult:
    natural: bool
    raw: str


def judge_naturalness(backend: Backend, document: str) -> QCResult:
    """Conservative yes/no naturalness verdict. Defaults to NATURAL (keep) when the
    judge fails to emit a parseable verdict — we never drop data on judge noise."""
    out = backend.generate(_QC_PROMPT.format(doc=document))
    m = _VERDICT.search(out)
    natural = (m.group(1).lower() == "natural") if m else True  # keep on ambiguity
    return QCResult(natural=natural, raw=out.strip()[:200])
