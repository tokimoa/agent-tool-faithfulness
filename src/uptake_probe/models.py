"""Model backends.

`Backend` is the model-agnostic interface the scaffolds talk to: a single
``generate(prompt) -> str``. Two implementations:

* ``MockModel`` — a deterministic, purely text-driven SIMULATOR. It reads the
  same prompt a real model sees and decides what to "use" via seeded draws whose
  probabilities encode the known stress effects (position / count / agentic
  omission / distraction). It exists ONLY to validate the measurement pipeline
  end-to-end and to power-analyse the harness on CPU. It produces NO scientific
  finding — real results require ``MLXModel`` (or an API backend).

* ``MLXModel`` — the real Apple-Silicon local backend (lazy mlx_lm import).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .schema import SOURCE_AUTHORITATIVE
from .util import stable_hash


@runtime_checkable
class Backend(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...

    def count_tokens(self, text: str) -> int: ...


def _approx_tokens(text: str) -> int:
    # ~0.75 words/token heuristic; good enough for relative cost estimates.
    return max(1, round(len(text.split()) / 0.75))


# --- shared parsing of the fixed render format ------------------------------

_VAL = r"[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}"
_P_TARGET = re.compile(r"What is the (.+?) for the (.+?)\?")
_P_FACT = re.compile(r"The (.+?) for the (.+?) is (" + _VAL + r")")
_P_HEADER = re.compile(r"\[(\w+)\] \((\w+)\)")
_P_SNIPPET = re.compile(
    r"\[(\w+)\] \((\w+)\) (.+?) — (.+?) (?:on record|\(unverified\))"
)


@dataclass(frozen=True)
class _Parsed:
    result_id: str
    source: str
    attribute: str
    entity: str
    value: str
    position: int


def _targets(prompt: str) -> list[tuple[str, str]]:
    """(attribute, entity) pairs asked for in the question."""
    return [(a.strip(), e.strip()) for a, e in _P_TARGET.findall(prompt)]


def _parse_result_chunks(block: str) -> list[_Parsed]:
    chunks = [c for c in re.split(r"\n\s*\n", block) if "[" in c]
    out: list[_Parsed] = []
    for pos, chunk in enumerate(chunks):
        head = _P_HEADER.search(chunk)
        fact = _P_FACT.search(chunk)
        if head and fact:
            out.append(_Parsed(head.group(1), head.group(2),
                               fact.group(1).strip(), fact.group(2).strip(),
                               fact.group(3), pos))
    return out


def _parse_listing(text: str) -> list[tuple[str, str, str, str, int]]:
    """(id, source, entity, attribute, position) from a SEARCH listing."""
    out = []
    for pos, m in enumerate(_P_SNIPPET.finditer(text)):
        out.append((m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip(), pos))
    return out


# --- the simulator -----------------------------------------------------------

@dataclass
class MockParams:
    base: float = 0.96            # uptake at an edge position, K=1
    c_center: float = 0.45        # penalty for being at the middle
    c_k: float = 0.025            # penalty per extra co-delivered result
    fetch_base: float = 0.90      # agentic: probability of fetching a needed result
    p_use_given_read: float = 0.90  # answer | (read/fetched) — the rest is Neglect
    p_distract: float = 0.35      # follow a contradictory value if it was observed
    p_peek: float = 0.55          # agentic: probability of also fetching a contradictory result
    floor: float = 0.05
    ceil: float = 0.99


def _centrality(pos: int, k: int) -> float:
    if k <= 1:
        return 0.0
    return 1.0 - 2.0 * abs(pos / (k - 1) - 0.5)


class MockModel:
    """Deterministic text-driven simulator (pipeline validation only)."""

    def __init__(self, model_seed: int = 0, params: MockParams | None = None,
                 name: str = "mock"):
        self.model_seed = model_seed
        self.params = params or MockParams()
        self.name = name

    # -- seeded decisions ----------------------------------------------------
    def _draw(self, *key: object) -> float:
        return random.Random(stable_hash(self.model_seed, *key)).random()

    def _p_read(self, entity: str, pos: int, k: int) -> float:
        p = self.params
        val = p.base - p.c_center * _centrality(pos, k) - p.c_k * (k - 1)
        return min(p.ceil, max(p.floor, val))

    def count_tokens(self, text: str) -> int:
        return _approx_tokens(text)

    # -- dispatch ------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        if "AVAILABLE ACTIONS:" in prompt:
            return self._agentic_action(prompt)
        return self._passive_answer(prompt)

    def logprob(self, prefix: str, continuation: str) -> float:
        """Deterministic stand-in for the real teacher-forced score, ONLY to
        self-test the abstention-margin probe's plumbing on CPU. It encodes the
        entity-anchored-copy prior: a code already present on the page is "cheap"
        (high logprob); abstaining is cheap only when the page carries no code at
        all. Produces NO scientific value — real margins require ``MLXModel``."""
        c = continuation.strip()
        page_has_code = re.search(_VAL, prefix) is not None
        if re.fullmatch(_VAL, c):                 # committing a concrete code
            base = -0.2 if c in prefix else -3.0  # copy on-page value vs invent one
        else:                                     # abstaining ("not available")
            base = -0.3 if not page_has_code else -2.0
        jitter = (stable_hash(self.model_seed, prefix[-32:], c) % 100) / 1000.0
        return base - jitter

    # -- passive -------------------------------------------------------------
    def _passive_answer(self, prompt: str) -> str:
        _, _, block = prompt.partition("RETRIEVED RESULTS:")
        parsed = _parse_result_chunks(block)
        k = len(parsed)
        by_entity: dict[str, list[_Parsed]] = {}
        for p in parsed:
            by_entity.setdefault(p.entity, []).append(p)

        trace_bits: list[str] = []
        answer_bits: list[str] = []
        for _attr, entity in _targets(prompt):
            recs = by_entity.get(entity, [])
            nec = next((r for r in recs if r.source == SOURCE_AUTHORITATIVE), None)
            contra = next((r for r in recs
                           if r.source != SOURCE_AUTHORITATIVE
                           and (nec is None or r.value != nec.value)), None)
            if nec is None:
                continue
            read = self._draw(entity, "read") < self._p_read(entity, nec.position, k)
            if read:
                trace_bits.append(f"{entity}: {nec.value}")
            if contra is not None and self._draw(entity, "distract") < self.params.p_distract:
                answer_bits.append(f"{entity} = {contra.value}")  # followed the lie
            elif read and self._draw(entity, "use") < self.params.p_use_given_read:
                answer_bits.append(f"{entity} = {nec.value}")
        trace = "Reasoning: " + ("; ".join(trace_bits) if trace_bits else "(no notes)")
        answer = "ANSWER: " + ("; ".join(answer_bits) if answer_bits else "(unknown)")
        return trace + "\n" + answer

    # -- agentic -------------------------------------------------------------
    def _fetch_plan(self, prompt: str, listing) -> list[str]:
        targets = {e for _a, e in _targets(prompt)}
        k = len(listing)
        plan: list[str] = []
        for rid, source, entity, _attr, pos in listing:
            if entity not in targets:
                continue
            if source == SOURCE_AUTHORITATIVE:
                if self._draw(entity, "fetch") < min(
                    self.params.ceil,
                    max(self.params.floor,
                        self.params.fetch_base
                        - self.params.c_center * _centrality(pos, k)
                        - self.params.c_k * (k - 1)),
                ):
                    plan.append(rid)
            else:  # a (possibly contradictory) unverified result on a target entity
                if self._draw(entity, "peek") < self.params.p_peek:
                    plan.append(rid)
        return plan

    def _agentic_action(self, prompt: str) -> str:
        _, _, transcript = prompt.partition("TRANSCRIPT:")
        if "Action: SEARCH" not in transcript:
            return "Action: SEARCH[all]"

        listing = _parse_listing(transcript)
        plan = self._fetch_plan(prompt, listing)
        already = set(re.findall(r"Action: FETCH\[(\w+)\]", transcript))
        for rid in plan:
            if rid not in already:
                return f"Action: FETCH[{rid}]"

        # all planned fetches done -> answer from observed full-text chunks
        observed = _parse_result_chunks(transcript)
        by_entity: dict[str, list[_Parsed]] = {}
        for p in observed:
            by_entity.setdefault(p.entity, []).append(p)
        answer_bits: list[str] = []
        for _attr, entity in _targets(prompt):
            recs = by_entity.get(entity, [])
            nec = next((r for r in recs if r.source == SOURCE_AUTHORITATIVE), None)
            contra = next((r for r in recs
                           if r.source != SOURCE_AUTHORITATIVE
                           and (nec is None or r.value != nec.value)), None)
            if contra is not None and self._draw(entity, "distract") < self.params.p_distract:
                answer_bits.append(f"{entity} = {contra.value}")
            elif nec is not None and self._draw(entity, "use") < self.params.p_use_given_read:
                answer_bits.append(f"{entity} = {nec.value}")
        return "Action: ANSWER[" + ("; ".join(answer_bits) if answer_bits else "unknown") + "]"


class MLXModel:
    """Real local backend via mlx_lm (lazy import; Apple Silicon).

    ``enable_thinking=False`` is the default: for a ReAct agent we want direct
    ``Action: ...`` output, and disabling Qwen/Gemma "thinking" both speeds up
    generation and improves protocol compliance. Falls back silently for models
    whose chat template doesn't expose the flag.
    """

    def __init__(self, model_id: str, max_tokens: int = 1024, temperature: float = 0.0,
                 enable_thinking: bool = False, name: str | None = None):
        # NOTE: keep max_tokens generous. Passive delivery bundles reasoning AND
        # the (possibly long, multi-value) answer in ONE generation, so a small
        # budget truncates the answer and undercounts uptake — an asymmetric
        # confound vs agentic (whose final ANSWER turn has the full budget). For
        # N necessary values budget ~ N*20 + slack; use >=1536 for N>=16.
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "MLXModel requires mlx_lm. Install with: uv sync --extra mlx"
            ) from exc
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.name = name or model_id.split("/")[-1]
        self._model, self._tokenizer = load(model_id)

    def count_tokens(self, text: str) -> int:  # pragma: no cover - needs mlx
        return len(self._tokenizer.encode(text))

    def _render(self, prompt: str) -> str:  # pragma: no cover - needs mlx
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def generate(self, prompt: str) -> str:  # pragma: no cover - needs mlx
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=self.temperature)
        return mlx_generate(
            self._model, self._tokenizer, prompt=self._render(prompt),
            max_tokens=self.max_tokens, sampler=sampler, verbose=False,
        )

    def logprob(self, prefix: str, continuation: str) -> float:  # pragma: no cover - needs mlx
        """Mean teacher-forced log-prob per token the model assigns to
        ``continuation`` immediately after ``prefix``. ``prefix`` is rendered
        through the chat template (user turn + assistant generation prompt) and
        ``continuation`` is scored as the start of the assistant reply. Used by the
        abstention-margin probe to read the model's INTERNAL preference between
        committing a value and abstaining, with no sampling."""
        import mlx.core as mx

        pre = self._render(prefix)
        pre_ids = self._tokenizer.encode(pre)
        full_ids = self._tokenizer.encode(pre + continuation)
        if full_ids[: len(pre_ids)] != pre_ids:  # boundary re-tokenised: fall back
            cont_ids = self._tokenizer.encode(continuation)
            if cont_ids and pre_ids and cont_ids[0] == pre_ids[0]:
                cont_ids = cont_ids[1:]  # drop a duplicated BOS
            full_ids = pre_ids + cont_ids
        tail = full_ids[len(pre_ids):]
        if not tail:
            return float("nan")
        logits = self._model(mx.array(full_ids)[None])[0]  # [T, V]
        total = 0.0
        for i, tok in enumerate(tail):
            row = logits[len(pre_ids) - 1 + i]             # predicts token `tok`
            total += float(row[tok] - mx.logsumexp(row))
        return total / len(tail)
