"""Tests for the benchmark generation pipeline (serialize + qc parsing).

These run WITHOUT a model: the QC judge backend is stubbed, so the pipeline logic
(parsing verdicts, conservative default) is verified deterministically.
"""

from uptake_probe.qc import judge_naturalness
from uptake_probe.schema import ResultQuality
from uptake_probe.serialize import instance_to_record
from uptake_probe.synth import make_instance


class _Stub:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt):
        return self.text

    def count_tokens(self, text):
        return len(text.split())


def test_serialize_record_has_judge_free_answer_key():
    inst = make_instance(seed=5, k=5, n_necessary=1,
                         necessary_quality=ResultQuality.ATTR_MISMATCH, style="realistic")
    rec = instance_to_record(inst, "M2_calibration", "attr_mismatch")
    assert rec["gold_values"] == inst.gold_values
    assert rec["decoys"] and all(isinstance(d, str) for d in rec["decoys"])
    assert rec["unobtainable_entities"]                      # bad result -> unobtainable
    assert rec["style"] == "realistic"
    # the released results expose only what an agent should see (no fact/quality leak)
    assert set(rec["results"][0]) == {"result_id", "snippet", "full_text",
                                      "source", "is_necessary"}


def test_qc_parses_verdicts_and_defaults_to_keep():
    assert judge_naturalness(_Stub("VERDICT: NATURAL"), "doc").natural is True
    assert judge_naturalness(_Stub("VERDICT: AWKWARD"), "doc").natural is False
    # unparseable -> conservative KEEP (never drop data on judge noise)
    assert judge_naturalness(_Stub("uh, maybe?"), "doc").natural is True


def test_good_condition_has_no_decoy_or_unobtainable():
    inst = make_instance(seed=5, k=5, n_necessary=1, necessary_quality=ResultQuality.GOOD)
    rec = instance_to_record(inst, "M2_calibration", "good")
    assert rec["decoys"] == []
    assert rec["unobtainable_entities"] == []
    assert rec["gold_values"][0] in rec["results"][inst.necessary_positions[0]]["full_text"]
