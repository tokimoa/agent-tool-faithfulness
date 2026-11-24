"""Eyeball check: print one instance + its agentic transcript, then a small grid.

    uv run python debug/check_uptake.py
"""

from __future__ import annotations

from uptake_probe.metrics import evaluate_run
from uptake_probe.models import MockModel
from uptake_probe.pilot import run_pilot, _print_table
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode
from uptake_probe.synth import make_instance


def show_one() -> None:
    inst = make_instance(seed=1000, k=4, n_necessary=1, position="center",
                         include_contradictory=True)
    print("QUESTION:", inst.question)
    print("\nNECESSARY:", inst.necessary_facts[0].entity,
          "->", inst.necessary_facts[0].value)
    print("RESULTS (position : necessary? : source):")
    for i, r in enumerate(inst.results):
        tag = "NECESSARY" if r.is_necessary else (r.distractor_type.value if r.distractor_type else "?")
        print(f"  [{i}] {r.result_id} {tag:<13} ({r.source}) :: {r.snippet}")

    backend = MockModel()
    rec = run(inst, backend, DeliveryMode.AGENTIC)
    print("\n--- AGENTIC TRANSCRIPT ---")
    print(rec.trace)
    print("--- ANSWER ---")
    print(rec.answer)
    print("observed_necessary:", rec.observed_necessary)
    for o in evaluate_run(inst, rec):
        print(f"  {o.entity}: used={o.used} OND={o.ond.value} "
              f"contra_present={o.contra_present} followed={o.contra_followed}")


def show_grid() -> None:
    rep = run_pilot(MockModel(), ks=(1, 4, 8, 12), modes=("passive", "agentic"),
                    n_instances=60, include_contradictory=False)
    _print_table(rep)


if __name__ == "__main__":
    show_one()
    show_grid()
