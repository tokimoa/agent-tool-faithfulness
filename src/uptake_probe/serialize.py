"""Serialize an ``Instance`` to a flat, HF-datasets-friendly JSON record.

The released record carries everything a consumer needs to (a) present the same
tool environment to their agent and (b) score judge-free, but NOT any LLM-judge
artifact — scoring is exact-match on the planted nonce values alone.
"""

from __future__ import annotations

from .schema import Instance


def instance_to_record(inst: Instance, module: str, condition: str) -> dict:
    """A flat dict ready for `datasets.Dataset.from_list` / JSONL export."""
    return {
        "instance_id": inst.instance_id,
        "module": module,            # M1_uptake / M2_calibration / ...
        "condition": condition,      # good / irrelevant / attr_mismatch / ...
        "style": inst.meta.get("style", "template"),
        "domain": inst.meta.get("domain", "access"),
        "question": inst.question,
        "k": inst.k,
        "n_necessary": inst.n_necessary,
        "position": inst.position,
        "seed": inst.seed,
        # the tool environment the agent will search/fetch over
        "results": [
            {
                "result_id": r.result_id,
                "snippet": r.snippet,
                "full_text": r.full_text,
                "source": r.source,
                "is_necessary": r.is_necessary,
            }
            for r in inst.results
        ],
        # --- judge-free answer key (the ONLY thing scoring needs) ---
        "gold_values": inst.gold_values,                 # exact-match target(s)
        "unobtainable_entities": sorted(inst.unobtainable_entities),
        "decoys": sorted({r.wrong_value for r in inst.results
                          if r.is_necessary and r.wrong_value}),
        "necessary_quality": inst.meta.get("necessary_quality", "good"),
        "mismatch_signal": inst.meta.get("mismatch_signal", "none"),
        "verify_entity": "verify the result is about the EXACT entity" in inst.question,
    }
