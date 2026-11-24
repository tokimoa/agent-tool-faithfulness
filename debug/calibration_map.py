"""Tool-result trust calibration: two symmetric error directions.

From the error-awareness run:
  false-acceptance (over-trust)  = IRRELEVANT confabulation rate
       (a bad/mismatched result is trusted and its value reported)
  false-rejection  (under-trust) = 1 - GOOD-ok
       (a valid result is rejected: the agent abstains/loops despite having the answer)

A well-calibrated agent is low on BOTH. The interesting claim: models miscalibrate
in DIFFERENT directions at similar sizes (training/family-driven, not parameter).

    uv run python debug/calibration_map.py
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    rows = json.loads(Path("results/error_awareness.json").read_text())
    by_model: dict = {}
    for r in rows:
        d = by_model.setdefault(r["model"], {"size": r["size_b"]})
        if r["cond"] == "irrelevant":
            d["fa"] = r["dist"].get("confabulate", 0.0)          # false acceptance / over-trust
        if r["cond"] == "good":
            d["fr"] = round(1.0 - r["dist"].get("correct", 0.0), 3)  # false rejection / under-trust

    def profile(fa, fr):
        hi = 0.30
        if fa >= hi and fr >= hi:
            return "both-miscalibrated"
        if fa >= hi:
            return "OVER-TRUST (confabulates)"
        if fr >= hi:
            return "UNDER-TRUST (over-rejects)"
        return "calibrated"

    items = sorted(by_model.items(), key=lambda kv: kv[1]["size"])
    print("TOOL-RESULT TRUST CALIBRATION  (two error directions)\n")
    print(f"{'model':<18}{'B':>5}{'over-trust(fa)':>15}{'under-trust(fr)':>16}  profile")
    print("-" * 78)
    for m, d in items:
        fa, fr = d.get("fa", float("nan")), d.get("fr", float("nan"))
        print(f"{m:<18}{d['size']:>5}{fa:>15.2f}{fr:>16.2f}  {profile(fa, fr)}")

    from collections import Counter
    profs = Counter(profile(d.get("fa", 0), d.get("fr", 0)) for _, d in items)
    print("\nprofile counts:", dict(profs))
    print("\nopposite miscalibrations at similar size (size cannot explain direction):")
    for m, d in items:
        if d.get("fa", 0) >= 0.30 or d.get("fr", 0) >= 0.30:
            print(f"  {m} ({d['size']}B): fa={d.get('fa',0):.2f} fr={d.get('fr',0):.2f}")


if __name__ == "__main__":
    main()
