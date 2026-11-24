"""Build the released benchmark dataset, end to end, FULLY LOCAL.

Pipeline (no API anywhere):
  1. GENERATE  — deterministic synthetic instances for every module/condition,
                 both template and realistic styles, over a fixed seed range.
  2. VALIDATE  — deterministic structural QC (validate.py): drop any instance
                 whose nonce/decoy/marker structure is wrong. Free, judge-free.
  3. NATURALNESS QC — a LOCAL LLM gates ONLY realistic-style docs for fluency
                 (qc.py). Conservative (keeps on ambiguity). Decisions are baked
                 into the released, filtered dataset; the judge never ships and is
                 never used at evaluation time.
  4. EXPORT    — HF-ready JSONL (frozen seed split) + a generator manifest (for
                 contamination-resistant regeneration) + a data card documenting
                 the judge-free scoring / generation-time-QC split and QC stats.

  uv run --extra mlx python debug/build_benchmark.py                 # full, local QC
  uv run python debug/build_benchmark.py --no-llm-qc                 # structural only
  uv run python debug/build_benchmark.py --no-llm-qc --n 5 --smoke   # tiny dry run

Output -> benchmark/   (dataset.jsonl, manifest.json, DATACARD.md, qc_log.jsonl)
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from uptake_probe.detect import detect_abstention, detect_value, has_value_code
from uptake_probe.domains import BUSINESS_DOMAINS
from uptake_probe.schema import ResultQuality
from uptake_probe.serialize import instance_to_record
from uptake_probe.synth import make_instance
from uptake_probe.validate import structural_issues

# The paper's input-ladder transforms, imported from the exact scripts that ran
# the experiments (deterministic string surgery on the necessary result).
sys.path.insert(0, str(Path(__file__).parent))
from input_cure import mutate as attr_mutate  # noqa: E402
from ladder_dims import mutate as dims_mutate  # noqa: E402

OUT = Path("benchmark")
# Chosen by debug/pick_qc_judge.py: top on the conservatism-weighted ranking
# (good→NATURAL 1.0, broken→AWKWARD 1.0, borderline→KEEP 1.0). The judge gates
# DATA NATURALNESS at generation time only; it never scores agent outputs, so its
# overlap with the benchmark roster is cosmetic (documented in the data card).
QC_JUDGE_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"

Q = ResultQuality
# module -> list of (condition_label, necessary_quality, mismatch_signal, verify)
MODULES: dict[str, list[tuple]] = {
    "M2_calibration": [
        ("good", Q.GOOD, "none", False),
        ("error", Q.ERROR, "none", False),
        ("empty", Q.EMPTY, "none", False),
        ("redacted", Q.REDACTED, "none", False),
        ("irrelevant", Q.IRRELEVANT, "none", False),
        ("irrelevant_near", Q.IRRELEVANT_NEAR, "none", False),
        ("irrelevant_far", Q.IRRELEVANT_FAR, "none", False),
        ("attr_mismatch", Q.ATTR_MISMATCH, "none", False),
        ("stale", Q.STALE, "none", False),
        ("partial", Q.PARTIAL, "none", False),
    ],
    "M3_curability": [
        ("irrelevant_verify", Q.IRRELEVANT, "none", True),
        ("irrelevant_sig_soft", Q.IRRELEVANT, "soft", False),
        ("irrelevant_sig_hard", Q.IRRELEVANT, "hard", False),
        ("good_verify", Q.GOOD, "none", True),
    ],
}
# M3b_interface: the paper's input-ladder transforms and label-ablation modes
# (record style, business domains only — matching the intervention studies).
# Each entry: (condition_label, base_quality, attr_label_mode, transform, lever).
# transform: None (generation knob only) | 'attr' (input_cure.mutate) |
#            'dims' (ladder_dims.mutate, takes a cond key).
M3B_BASE = 60000  # disjoint seed range (experiments: 7xxxxx/8xxxxx; M2/M3: 20000+)
M3B: list[tuple] = [
    ("attr_absence", Q.ATTR_MISMATCH, "wrong", "attr", "absence"),
    ("attr_tags", Q.ATTR_MISMATCH, "wrong", "attr", "tags"),
    ("attr_isolate", Q.ATTR_MISMATCH, "wrong", "attr", "isolate"),
    ("attr_label_neutral", Q.ATTR_MISMATCH, "neutral", None, None),
    ("attr_label_bare", Q.ATTR_MISMATCH, "bare", None, None),
    ("attr_label_warn", Q.ATTR_MISMATCH, "warn", None, None),
    ("entity_absence", Q.IRRELEVANT, None, "dims", "absence"),
    ("entity_isolate", Q.IRRELEVANT, None, "dims", "isolate"),
    ("stale_absence", Q.STALE, None, "dims", "absence"),
    ("stale_isolate", Q.STALE, None, "dims", "isolate"),
    ("good_isolate", Q.GOOD, None, "dims", "isolate"),
]
_DIMS_KEY = {Q.IRRELEVANT: "irrelevant", Q.STALE: "stale", Q.GOOD: "good"}

# record carriers only differ for these qualities; for the rest record==template
_REALISTIC_QUALITIES = {Q.GOOD, Q.IRRELEVANT, Q.IRRELEVANT_NEAR, Q.IRRELEVANT_FAR,
                        Q.ATTR_MISMATCH, Q.STALE, Q.PARTIAL}
# generation grid:
#   - template / access  : the abstract control (one per condition)
#   - record   / <biz>   : multi-field business-domain tool outputs (the asked field
#                          is one of several; prose drawn from the validated, diverse
#                          frame pool — see debug/gen_text_pool.py)
_VARIANTS = [("template", "access")] + [("record", d) for d in BUSINESS_DOMAINS]


def generate(n: int, k: int):
    """Yield (module, condition, style, domain, instance) for the whole grid.

    Each (condition, style, domain) cell gets a DISJOINT seed block (stride 1000),
    so no nonce value is shared across cells — every record carries a distinct
    planted value (otherwise reusing seed=base+i across cells collides values,
    since the value draw lands on the same RNG state in different cells)."""
    base = 20000  # disjoint from any experiment seed range
    cell = 0
    for module, conds in MODULES.items():
        for clabel, quality, signal, verify in conds:
            for style, domain in _VARIANTS:
                if style == "record" and quality not in _REALISTIC_QUALITIES:
                    continue  # no distinct record form -> skip the duplicate
                cell_base = base + cell * 1000
                cell += 1
                for i in range(n):
                    inst = make_instance(seed=cell_base + i, k=k, n_necessary=1,
                                         position="center", necessary_quality=quality,
                                         mismatch_signal=signal, verify_entity=verify,
                                         style=style, domain=domain)
                    yield module, clabel, style, domain, inst


def _nec(inst):
    return next(r for r in inst.results if r.is_necessary)


def transform_issues(base, mutated, clabel, lever) -> list[str]:
    """Post-transform structural checks. The BASE instance already passed the
    full structural_issues gate; the transform is deterministic string surgery,
    so we verify only what the surgery must (and must not) have done."""
    issues: list[str] = []
    nb, nm = _nec(base), _nec(mutated)
    fact = nb.fact
    if detect_value(fact.value, nm.full_text) != (nb.quality is Q.GOOD):
        issues.append(f"{clabel}: gold-value presence wrong after transform")
    if lever == "isolate" and nb.quality is not Q.GOOD:
        if has_value_code(nm.full_text):
            issues.append(f"{clabel}: isolate left a value-code in the doc")
    elif lever == "isolate" and nb.quality is Q.GOOD:
        if not detect_value(fact.value, nm.full_text):
            issues.append(f"{clabel}: good_isolate lost the gold value")
    elif lever in ("absence", "tags"):
        if nb.wrong_value and not detect_value(nb.wrong_value, nm.full_text):
            issues.append(f"{clabel}: {lever} removed the decoy (must stay)")
        if len(nm.full_text) <= len(nb.full_text):
            issues.append(f"{clabel}: {lever} added no annotation")
        if nb.quality is Q.STALE and not detect_abstention(nm.full_text):
            issues.append(f"{clabel}: stale marker lost")
    return issues


def label_mode_issues(inst) -> list[str]:
    """Structural gate for attr_label_mode variants: the full ATTR_MISMATCH check
    minus the asked-attribute-leak clause (the warn label deliberately names the
    requested field to contradict the decoy)."""
    return [i for i in structural_issues(inst) if "leaks asked attribute" not in i]


def generate_m3b(n: int, k: int):
    """Yield (condition_label, style, domain, base, mutated) for the M3b grid."""
    for ci, (clabel, quality, label_mode, transform, lever) in enumerate(M3B):
        for di, dom in enumerate(BUSINESS_DOMAINS):
            cell_base = M3B_BASE + (ci * len(BUSINESS_DOMAINS) + di) * 1000
            for i in range(n):
                kwargs = dict(seed=cell_base + i, k=k, n_necessary=1,
                              position="center", necessary_quality=quality,
                              style="record", domain=dom)
                if label_mode:
                    kwargs["attr_label_mode"] = label_mode
                base = make_instance(**kwargs)
                if transform == "attr":
                    mutated = attr_mutate(base, lever)
                elif transform == "dims":
                    mutated = dims_mutate(base, _DIMS_KEY[quality], lever)
                else:
                    mutated = base
                yield clabel, "record", dom, base, mutated, transform, lever


def main() -> None:
    args = sys.argv[1:]
    no_llm = "--no-llm-qc" in args
    smoke = "--smoke" in args
    n = 6 if smoke else (int(args[args.index("--n") + 1]) if "--n" in args else 50)
    k = 5
    # the per-doc naturalness judge. Default differs from the pool author on purpose:
    # the frame pool is written by gemma-4-26b, so judging with a DIFFERENT model
    # (Qwen3.5-27B) avoids a generator-grades-itself circularity.
    qc_judge = args[args.index("--qc-judge") + 1] if "--qc-judge" in args else QC_JUDGE_ID
    OUT.mkdir(exist_ok=True)

    # 1+2: generate + deterministic structural validation
    kept, dropped_struct = [], 0
    realistic_docs = []   # (idx-in-kept, necessary full_text) for the LLM QC pass
    for module, clabel, style, domain, inst in generate(n, k):
        issues = structural_issues(inst)
        if issues:
            dropped_struct += 1
            continue
        rec = instance_to_record(inst, module, clabel)
        rec["_qc_natural"] = True   # default; record docs may be overwritten below
        if style == "record":
            nec = next(r for r in inst.results if r.is_necessary)
            realistic_docs.append((len(kept), nec.full_text))
        kept.append(rec)

    # 1b+2b: M3b interface-transform cells (paper's input ladder + label ablation)
    for clabel, style, domain, base, mutated, transform, lever in generate_m3b(n, k):
        if transform is None:
            issues = label_mode_issues(mutated)
        else:
            issues = structural_issues(base) + transform_issues(base, mutated,
                                                                clabel, lever)
        if issues:
            dropped_struct += 1
            continue
        rec = instance_to_record(mutated, "M3b_interface", clabel)
        if lever:
            rec["instance_id"] = f"{rec['instance_id']}-{lever}"
        rec["_qc_natural"] = True
        realistic_docs.append((len(kept), _nec(mutated).full_text))
        kept.append(rec)
    print(f"generated+validated: kept {len(kept)}, dropped(structural) {dropped_struct}")

    # 3: LOCAL LLM naturalness QC on realistic docs only
    qc_log = []
    dropped_qc = 0
    if not no_llm and realistic_docs:
        from uptake_probe.models import MLXModel
        from uptake_probe.qc import judge_naturalness
        print(f"QC judge (local): {qc_judge} on {len(realistic_docs)} record docs")
        be = MLXModel(qc_judge, max_tokens=64)
        for idx, doc in realistic_docs:
            res = judge_naturalness(be, doc)
            qc_log.append({"instance_id": kept[idx]["instance_id"],
                           "natural": res.natural, "raw": res.raw})
            if not res.natural:
                kept[idx]["_qc_natural"] = False
                dropped_qc += 1
        del be
        gc.collect()
        kept = [r for r in kept if r.pop("_qc_natural", True)]
        print(f"LLM-QC: dropped {dropped_qc} awkward realistic docs")
    else:
        for r in kept:
            r.pop("_qc_natural", None)

    # 4: export
    (OUT / "dataset.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
    if qc_log:
        (OUT / "qc_log.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in qc_log) + "\n")

    by_mod = Counter(r["module"] for r in kept)
    by_cond = Counter((r["module"], r["condition"], r["style"], r["domain"]) for r in kept)
    by_dom = Counter(r["domain"] for r in kept)
    manifest = {
        "n_per_cell": n, "k": k, "seed_base": 20000, "m3b_seed_base": M3B_BASE,
        "modules": {m: sorted({r["condition"] for r in kept if r["module"] == m})
                    for m in by_mod},
        "domains": dict(sorted(by_dom.items())),
        "qc_judge": None if no_llm else qc_judge,
        "frame_pool_author": QC_JUDGE_ID,  # gemma-4-26b authored the validated frame pool
        "qc_dropped": dropped_qc, "structural_dropped": dropped_struct,
        "total_instances": len(kept),
        "scoring": "judge-free exact-match on gold_values; abstention via regex "
                   "(human-validated kappa=0.93 development, 0.94 held-out). "
                   "No LLM at evaluation time.",
        "regenerate": "uptake_probe.synth.make_instance(seed=20000+i, ...) reproduces "
                      "every M2/M3 instance; M3b applies the named deterministic "
                      "transform (debug/input_cure.py or debug/ladder_dims.py mutate) "
                      "to make_instance(seed=60000+i, ...). Use a fresh seed_base for "
                      "a contamination-free pull.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    _write_datacard(manifest, by_mod, by_cond)
    print(f"\nsaved -> {OUT}/dataset.jsonl ({len(kept)} instances), manifest.json, DATACARD.md")


def _write_datacard(manifest, by_mod, by_cond) -> None:
    lines = [
        "# Tool-Result Reliance — Diagnostic Benchmark (data card)", "",
        "**Reference-free (judge-free).** Scoring is exact-match on planted, "
        "un-memorizable nonce values plus a human-validated abstention regex "
        "(kappa=0.93 development, 0.94 held-out). **No LLM is used at evaluation "
        "time.** An LLM was used only at *generation* time as a naturalness gate on "
        "realistic-prose documents; its decisions are baked into this filtered "
        "release and it is not shipped.", "",
        "Terminology note: this card and the field/module names predate the paper's "
        "final vocabulary. Mapping: *confabulation* (fields) = **silent "
        "mis-grounding** (paper); *calibration* (module `M2_calibration`) = "
        "**reliance behaviour**; *curability* (modules `M3_curability`/"
        "`M3b_interface`) = **mitigation**. Module IDs are kept stable for "
        "reproducibility. All instances are in **English**.", "",
        f"- instances: **{manifest['total_instances']}** "
        f"(n={manifest['n_per_cell']}/cell, k={manifest['k']})",
        f"- generation-time QC judge (local): `{manifest['qc_judge']}` "
        f"(dropped {manifest['qc_dropped']} awkward; structural drops "
        f"{manifest['structural_dropped']}). This judge gates document *naturalness* "
        f"only; it never scores agent outputs, so its appearance in the model roster "
        f"is not a circularity.",
        f"- contamination resistance: nonce values are random; regenerate with a "
        f"fresh seed for an uncontaminated pull. {manifest['regenerate']}", "",
        f"- domains: {manifest['domains']} "
        f"(template=abstract control on 'access'; record=multi-field business tool "
        f"outputs across {len([d for d in manifest['domains'] if d != 'access'])} "
        f"domains, asked field is one of several, prose drawn from a validated diverse "
        f"frame pool — values stay nonce codes so scoring is unchanged).", "",
        "## Coverage vs. the paper", "",
        "`M2_calibration` = the paper's Table 2 grid (good + three entity variants + "
        "attr-mismatch + stale + partial across five domains, plus explicit failures "
        "error/empty/redacted). `M3b_interface` = the paper's input-ladder transforms "
        "(absence/tags/isolate on attribute; absence/isolate on entity and recency; "
        "the GOOD-side isolate control) and the label-ablation modes "
        "(neutral/bare/warn; `wrong` = plain attr_mismatch in M2). `M3_curability` = "
        "legacy signal/verify variants from an earlier intervention design (the "
        "good_verify/irrelevant_verify cells back the paper's over-abstention and "
        "verify-cue analyses). Prompt-level interventions (cite/verify/consequence), "
        "the commit scaffold, and the multi-agent verification protocol are runner-"
        "side (no document change): see `debug/mitigation_contrast.py`, "
        "`debug/commit_contrast.py`, `debug/ma_dimensions_record.py`; their "
        "per-instance raw outputs ship in `results/raw/`.", "",
        "## License", "",
        "- Data (this file): **CC BY 4.0**.",
        "- Code (generator / scorer / harness): **MIT** (see `LICENSE`).", "",
        "## Modules / conditions / domains", "",
        "| module | condition | style | domain | n |", "|---|---|---|---|---|",
    ]
    for (m, c, s, d), cnt in sorted(by_cond.items()):
        lines.append(f"| {m} | {c} | {s} | {d} | {cnt} |")
    lines += ["", "## Scoring (consumer-side)", "",
              "For each necessary fact: the agent's answer USES the result iff a "
              "`gold_values` entry appears verbatim (separator-tolerant). On "
              "unobtainable conditions (`unobtainable_entities` non-empty) any "
              "`decoys` value asserted without an abstention marker is a "
              "*silent mis-grounding* (field name: confabulation). See "
              "`uptake_probe.detect` / `uptake_probe.metrics` for the reference "
              "scorer."]
    (OUT / "DATACARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
