"""Generate the LaTeX tables added in the camera-ready revision.

Reads results/outcome_split.json (debug/outcome_split.py) and
results/fallible_detector.json (debug/fallible_detector.py) and writes the table
bodies to paper/tables/*.tex, which main.tex \\input's. No number in those
tables is typed by hand.

  uv run python debug/outcome_split.py && uv run python debug/tables_camera_ready.py
"""

from __future__ import annotations

import json
from pathlib import Path

from uptake_probe.stats import wilson_ci

OS = json.loads(Path("results/outcome_split.json").read_text())
FD = json.loads(Path("results/fallible_detector.json").read_text())
OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Phi-4-mini", "aya-expanse-8b", "gemma-3-12b", "Phi-4", "Mistral-Small-24B",
          "gemma-4-26b", "Qwen3.5-27B", "Qwen3.6-35B"]
CORE6 = [m for m in MODELS if m not in ("Phi-4", "Mistral-Small-24B")]
CATS = ("asserted", "caveated", "abstain", "noncode", "none")
DIMS = [("irrelevant", "entity"), ("attr_mismatch", "attribute"), ("stale", "recency"),
        ("partial", "completeness")]


def f2(x):
    return f"{x:.2f}"


def s2(x):
    """compact: .82 / 1.0"""
    s = f"{x:.2f}"
    return "1.0" if s == "1.00" else s[1:]


def rate(r, c):
    """Rate of outcome ``c`` from the stored counts, so that a rate already
    rounded to three decimals in the JSON is not rounded a second time."""
    return r[c + "_k"] / r["n"] if r.get("n") and (c + "_k") in r else r[c]


def ci(k, n):
    lo, hi = wilson_ci(k, n)
    return f"{k / n:.2f}\\,[{s2(lo)},{s2(hi)}]"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / name} ({len(lines)} rows)")


# --- body: pooled five-way split on the baseline grid ------------------------
rows = []
for cond, dim in DIMS:
    r = OS["baseline"]["pooled"][f"{cond}|orig"]
    exposed = (r["asserted_k"] + r["caveated_k"]) / r["n"]
    rows.append(f"{dim} & " + " & ".join(f2(rate(r, c)) for c in CATS) + f" & {f2(exposed)} \\\\")
write("outcome_pooled.tex", rows)

# --- appendix: per-model five-way split on the baseline grid -----------------
rows = []
for m in MODELS:
    cells = []
    for cond, _ in DIMS[:3]:
        r = OS["baseline"]["per_model"][f"{m}|{cond}|orig"]
        cells += [s2(rate(r, c)) for c in CATS]
    rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule")
cells = []
for cond, _ in DIMS[:3]:
    r = OS["baseline"]["pooled"][f"{cond}|orig"]
    cells += [s2(rate(r, c)) for c in CATS]
rows.append("pooled & " + " & ".join(cells) + " \\\\")
write("outcome_per_model.tex", rows)

# --- appendix: pooled five-way split for every intervention study ------------
STUDIES = [
    ("Baseline cues (entity; 8 models)", "baseline", "irrelevant",
     [("orig", "orig", "irrelevant|orig"), ("question cue", None, "irrelevant_verify|orig"),
      ("result cue, soft", None, "irrelevant_sig_soft|orig"),
      ("result cue, hard", None, "irrelevant_sig_hard|orig")]),
    ("Label ablation (attribute; 8 models)", "label_ablation", "attr_mismatch",
     [(a, a, None) for a in ("wrong", "neutral", "bare", "warn")]),
    ("Prompt mitigation, attribute (core-6)", "prompt_mitigation", "attr_mismatch",
     [(a, k, None) for a, k in (("orig", "orig"), ("cite", "cite"), ("verify", "verify"),
                                ("conseq.", "consequence"))]),
    ("Prompt mitigation, entity (core-6)", "prompt_mitigation", "irrelevant",
     [(a, k, None) for a, k in (("orig", "orig"), ("cite", "cite"), ("verify", "verify"),
                                ("conseq.", "consequence"))]),
    ("Prompt mitigation, recency (core-6)", "prompt_mitigation", "stale",
     [(a, k, None) for a, k in (("orig", "orig"), ("cite", "cite"), ("verify", "verify"),
                                ("conseq.", "consequence"))]),
    ("Input ladder, attribute (8 models)", "ladder_attribute", "attr_mismatch",
     [(a, a, None) for a in ("orig", "cite", "absence", "tags", "isolate")]),
    ("Input ladder, entity (8 models)", "ladder_dims", "irrelevant",
     [(a, a, None) for a in ("orig", "absence", "isolate")]),
    ("Input ladder, recency (8 models)", "ladder_dims", "stale",
     [(a, a, None) for a in ("orig", "absence", "isolate")]),
]
rows = []
for title, study, cond, arms in STUDIES:
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\")
    for label, arm, key in arms:
        r = OS[study]["pooled"][key or f"{cond}|{arm}"]
        rows.append(f"\\quad {label} & {r['n']} & " + " & ".join(f2(rate(r, c)) for c in CATS) + " \\\\")
    rows.append("\\addlinespace[2pt]")
write("outcome_studies.tex", rows[:-1])

# --- appendix: commit study with invalid writes and the any-write rule --------
rows = []
for m in CORE6:
    cells = []
    for cond, _ in DIMS[:3]:
        rep = OS["commit"]["per_model"][f"{m}|{cond}|report"]
        com = OS["commit"]["per_model"][f"{m}|{cond}|commit"]
        cells += [f2(rep["asserted"]), f2(com["commit_code"]), f2(com["commit_invalid"]),
                  ci(com["strict_k"], com["n"])]
    rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule")
cells = []
for cond, _ in DIMS[:3]:
    rep = OS["commit"]["pooled"][f"{cond}|report"]
    com = OS["commit"]["pooled"][f"{cond}|commit"]
    cells += [f2(rep["asserted"]), f2(com["commit_code"]), f2(com["commit_invalid"]),
              ci(com["strict_k"], com["n"])]
rows.append("pooled & " + " & ".join(cells) + " \\\\")
write("commit_strict.tex", rows)

# --- body: fallible detectors -------------------------------------------------
NAMES = [("lexical", "rule-based"), ("gemma-4-26b", "gemma-4-26b"), ("Phi-4-mini", "Phi-4-mini")]
p0 = FD["lexical"]["pooled"]
rows = [f"none (\\emph{{orig}}) & -- & -- & {f2(p0['irrelevant']['orig'])} & "
        f"{f2(p0['attr_mismatch']['orig'])} & {f2(p0['stale']['orig'])} & "
        f"{f2(p0['good']['orig_util'])} \\\\"]
for key, label in NAMES:
    d = FD[key]
    fn = d["detector_rates"]
    assert fn["irrelevant"]["false_negative"] == 0 and fn["attr_mismatch"]["false_negative"] == 0
    p = d["pooled"]
    rows.append(f"{label} & {f2(fn['stale']['false_negative'])} & "
                f"{f2(fn['good']['false_positive'])} & {f2(p['irrelevant']['chained'])} & "
                f"{f2(p['attr_mismatch']['chained'])} & {f2(p['stale']['chained'])} & "
                f"{f2(p['good']['chained_util'])} \\\\")
rows.append("oracle (\\emph{isolate}) & 0 & 0 & 0.00 & 0.00 & 0.00 & "
            f"{f2(p0['good']['orig_util'])} \\\\")
write("fallible.tex", rows)

# --- appendix: fallible detectors, per agent model ---------------------------
rows = []
for m in MODELS:
    cells = []
    for key, _ in NAMES:
        c = FD[key]["chained"][m]
        cells += [f2(c["stale"]["chained"]), f2(c["good"]["chained_util"])]
    o = FD["lexical"]["chained"][m]
    rows.append(f"{m} & {f2(o['stale']['orig'])} & {f2(o['good']['orig_util'])} & "
                + " & ".join(cells) + " \\\\")
write("fallible_per_model.tex", rows)

# --- body: per-model input ladder, all dimensions (point estimates) ----------
IC = {(r["model"], r["lever"]): r for r in json.loads(Path("results/input_cure.json").read_text())}
LD = {(r["model"], r["cond"], r["lever"]): r
      for r in json.loads(Path("results/ladder_dims.json").read_text())}
RAW_IC = [json.loads(l) for l in Path("results/raw/input_cure_raw.jsonl").read_text().splitlines() if l.strip()]
RAW_LD = [json.loads(l) for l in Path("results/raw/ladder_dims_raw.jsonl").read_text().splitlines() if l.strip()]


def _rate(r):
    return r["confab"] if "confab" in r else r["util"]


def _pooled_ic(lever):
    xs = [r for r in RAW_IC if r["lever"] == lever and r["model"] in MODELS]
    return sum(r["outcome"] == "confab" for r in xs) / len(xs)


def _pooled_ld(cond, lever):
    xs = [r for r in RAW_LD if r["cond"] == cond and r["lever"] == lever and r["model"] in MODELS]
    hit = "used" if cond == "good" else "confab"
    return sum(r["outcome"] == hit for r in xs) / len(xs)


A_LEV = ("orig", "cite", "absence", "tags", "isolate")
D_LEV = ("orig", "absence", "isolate")
rows = []
for m in MODELS:
    cells = [f2(IC[(m, lv)]["confab"]) for lv in A_LEV]
    for cond in ("irrelevant", "stale"):
        cells += [f2(_rate(LD[(m, cond, lv)])) for lv in D_LEV]
    cells += [f2(_rate(LD[(m, "good", lv)])) for lv in ("orig", "isolate")]
    rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule")
cells = [f2(_pooled_ic(lv)) for lv in A_LEV]
for cond in ("irrelevant", "stale"):
    cells += [f2(_pooled_ld(cond, lv)) for lv in D_LEV]
cells += [f2(_pooled_ld("good", lv)) for lv in ("orig", "isolate")]
rows.append("pooled & " + " & ".join(cells) + " \\\\")
write("ladder_body.tex", rows)

for f in sorted(OUT.glob("*.tex")):
    print(f"\n% {f.name}\n" + f.read_text(), end="")
