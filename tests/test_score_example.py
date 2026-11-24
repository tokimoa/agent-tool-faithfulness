"""The offline answer scorer reproduces a published cell from the released logs."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))

from score_example import score_answers_file  # noqa: E402

DATASET = ROOT / "benchmark" / "dataset.jsonl"
RAW = ROOT / "results" / "raw" / "baseline_raw.jsonl"


@pytest.mark.skipif(not (DATASET.exists() and RAW.exists()), reason="released data not present")
def test_reproduces_baseline_profile():
    recs = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    counts = score_answers_file(recs, RAW, model="Phi-4-mini")
    stale = counts["stale"]
    mis = stale.get("confab_decoy", 0) + stale.get("confab_novel", 0)
    assert sum(stale.values()) == 125
    assert round(mis / 125, 2) == 0.67  # the paper's recency cell for Phi-4-mini
    good = counts["good"]
    assert round(good["used"] / sum(good.values()), 2) == 0.94
