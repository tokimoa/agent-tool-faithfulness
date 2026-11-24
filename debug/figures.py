"""Generate the paper figures from the saved result JSONs.

    uv run --extra viz python debug/figures.py

Writes PNGs to results/figs/. Each figure degrades gracefully if its source
JSON is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("results/figs")
FIGS.mkdir(parents=True, exist_ok=True)


def _load(name):
    p = Path("results") / name
    return json.loads(p.read_text()) if p.exists() else None


def fig1_calibration():
    rows = _load("error_awareness.json")
    if not rows:
        return
    by = {}
    for r in rows:
        d = by.setdefault(r["model"], {"size": r["size_b"]})
        if r["cond"] == "irrelevant":
            d["fa"] = r["dist"].get("confabulate", 0.0)
        if r["cond"] == "good":
            d["fr"] = 1.0 - r["dist"].get("correct", 0.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    for m, d in by.items():
        fa, fr = d.get("fa", 0), d.get("fr", 0)
        ax.scatter(fa, fr, s=40 + d["size"] * 6, alpha=0.7, color="#3366cc")
        ax.annotate(m, (fa, fr), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.3, ls="--", c="gray", lw=0.6)
    ax.axvline(0.3, ls="--", c="gray", lw=0.6)
    ax.set_xlabel("over-trust  (confabulation on a bad result)")
    ax.set_ylabel("under-trust  (rejection of a good result)")
    ax.set_title("Tool-result trust calibration (two error directions)\n"
                 "marker size ∝ params; quadrants = behavioural profiles")
    ax.set_xlim(-0.05, 0.95)
    ax.set_ylim(-0.05, 0.95)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_calibration.png", dpi=150)
    plt.close(fig)
    print("fig1_calibration.png")


def fig2_uptake_vs_size():
    data = _load("capacity_sweep_final.json")
    if not data:
        return
    pts = [(r["size_b"], r["coverage"], r["model"]) for r in data["rows"]
           if r["mode"] == "agentic"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, c, m in pts:
        ax.scatter(s, c, s=45, alpha=0.7, color="#cc3333")
        ax.annotate(m, (s, c), fontsize=6, xytext=(3, -2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("params (B, log)")
    ax.set_ylabel("agentic uptake coverage (N=8)")
    ax.set_title("Uptake is model-dependent: full 0–1 range within every size class\n"
                 "(Spearman ρ=0.64 with size, but model identity dominates)")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_uptake_vs_size.png", dpi=150)
    plt.close(fig)
    print("fig2_uptake_vs_size.png")


def fig3_explicit_vs_silent():
    rows = _load("error_awareness.json")
    if not rows:
        return
    by = {}
    for r in rows:
        d = by.setdefault(r["model"], {"size": r["size_b"]})
        d[r["cond"]] = r["dist"].get("confabulate", 0.0)
    items = sorted(by.items(), key=lambda kv: kv[1]["size"])
    labels = [m for m, _ in items]
    err = [d.get("error", 0) for _, d in items]
    irr = [d.get("irrelevant", 0) for _, d in items]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - 0.2 for i in x], err, 0.4, label="explicit failure (ERROR)", color="#888")
    ax.bar([i + 0.2 for i in x], irr, 0.4, label="silent mismatch (IRRELEVANT)", color="#cc3333")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("confabulation rate")
    ax.set_title("Agents handle EXPLICIT tool failures (≈0) but are fooled by "
                 "SILENT quality failures (0–0.85)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_explicit_vs_silent.png", dpi=150)
    plt.close(fig)
    print("fig3_explicit_vs_silent.png")


def fig4_catch_matrix():
    data = _load("ma2_matrix.json")
    if not data:
        return
    catch = data["catch"]
    sups = list(catch.keys())
    workers = list(next(iter(catch.values())).keys())
    mat = [[catch[s].get(w) if catch[s].get(w) is not None else float("nan")
            for w in workers] for s in sups]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(workers)))
    ax.set_xticklabels(workers, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(sups)))
    ax.set_yticklabels(sups, fontsize=8)
    for i, s in enumerate(sups):
        for j, w in enumerate(workers):
            v = catch[s].get(w)
            if v is not None:
                ax.text(j, i, f"{v:.2f}" + ("*" if s == w else ""),
                        ha="center", va="center", fontsize=8)
    ax.set_xlabel("worker (whose confabulation)")
    ax.set_ylabel("supervisor (verifier)")
    ax.set_title("Catch rate: verification efficacy is a calibration property\n"
                 "(* = self; Qwen3.5-9B row catches all; weak verifiers miss hard confabs)")
    fig.colorbar(im, ax=ax, label="catch rate")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_catch_matrix.png", dpi=150)
    plt.close(fig)
    print("fig4_catch_matrix.png")


_MECH_COLORS = {
    "Phi-4-mini": "#d62728", "aya-expanse-8b": "#ff7f0e", "Qwen3.5-9B": "#2ca02c",
    "gemma-3-12b": "#9467bd", "Mistral-Small-24B": "#1f77b4",
    # larger / newer models added for scale & generational robustness
    "gemma-4-26b": "#8c564b", "Qwen3.5-27B": "#17becf", "Qwen3.6-35B": "#bcbd22",
    "Qwen3-32B": "#7f7f7f", "Phi-4": "#e377c2", "Yi-1.5-34B": "#393b79",
}


def _mech_color(model):
    return _MECH_COLORS.get(model, "#555555")


def fig5_plausibility_gradient():
    rows = _load("mechanism.json")
    if not rows:
        return
    order = ["irrelevant_near", "irrelevant", "irrelevant_far"]
    xlabels = ["NEAR\n(near-identical entity)", "MID\n(different entity)", "FAR\n(different attr+entity)"]
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["cond"]] = r["confab_rate"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, d in sorted(by.items()):
        ys = [d.get(c, float("nan")) for c in order]
        ax.plot(range(len(order)), ys, "-o", color=_mech_color(m), label=m, lw=2, ms=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("confabulation rate (silent mismatch)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Confabulation tracks the PLAUSIBILITY of the wrong result\n"
                 "(falls toward FAR ⇒ the agent skips the entity-identity check)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_plausibility_gradient.png", dpi=150)
    plt.close(fig)
    print("fig5_plausibility_gradient.png")


def fig6_verify_intervention():
    rows = _load("mechanism_c2.json")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    # dodge labels that land on (near-)identical endpoints so they stay legible
    used: list[float] = []
    for r in sorted(rows, key=lambda r: r["confab_on"]):
        off, on = r["confab_off"], r["confab_on"]
        c = _mech_color(r["model"])
        ax.plot([0, 1], [off, on], "-o", color=c, lw=2, ms=7)
        label_y = on
        while any(abs(label_y - u) < 0.04 for u in used):
            label_y += 0.045
        used.append(label_y)
        ax.text(1.04, label_y, r["model"], fontsize=7, va="center", color=c)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no instruction", '"verify the entity"'])
    ax.set_xlim(-0.15, 1.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("confabulation rate (silent mismatch)")
    ax.set_title("An explicit verify-the-entity instruction only PARTIALLY mitigates\n"
                 "(gemma-3-12b ignores it entirely ⇒ prompt-resistant grounding failure)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_verify_intervention.png", dpi=150)
    plt.close(fig)
    print("fig6_verify_intervention.png")


def fig7_lexical_signal():
    rows = _load("lexical_signal.json")
    if not rows:
        return
    order = ["none", "soft", "hard"]
    xlabels = ["none\n(silent id mismatch)", "soft\n(\"related but distinct\")",
               "hard\n(\"no exact match\")"]
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["signal"]] = r["confab_rate"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, d in sorted(by.items()):
        ys = [d.get(s, float("nan")) for s in order]
        ax.plot(range(len(order)), ys, "-o", color=_mech_color(m), label=m, lw=2, ms=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("confabulation rate (silent mismatch)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Adding an explicit mismatch WORDING to the result\n"
                 "(drop ⇒ string-driven; flat-high ⇒ ignores even a stated warning)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_lexical_signal.png", dpi=150)
    plt.close(fig)
    print("fig7_lexical_signal.png")


def fig8_taxonomy():
    # prefer the hardened run (n=50 + Wilson CIs); fall back to the n=20 point probe
    hardened = _load("taxonomy_hardened.json")
    rows = hardened or _load("taxonomy.json")
    if not rows:
        return
    order = ["irrelevant", "attr_mismatch", "stale", "partial"]
    xlabels = ["irrelevant\n(wrong entity)", "attr-mismatch\n(wrong attribute)",
               "stale\n(superseded value)", "partial\n(truncated value)"]
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["cond"]] = r
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    n_models = len(by)
    # tiny x-jitter so overlapping CIs/markers stay readable
    for mi, (m, d) in enumerate(sorted(by.items())):
        dx = (mi - n_models / 2) * 0.012
        xs = [i + dx for i in range(len(order))]
        ys = [d[c]["confab_rate"] if c in d else float("nan") for c in order]
        if hardened:
            lo = [d[c]["confab_rate"] - d[c]["wilson_lo"] if c in d else 0 for c in order]
            hi = [d[c]["wilson_hi"] - d[c]["confab_rate"] if c in d else 0 for c in order]
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", color=_mech_color(m), label=m,
                        lw=1.6, ms=5, capsize=2, elinewidth=1, alpha=0.9)
        else:
            ax.plot(xs, ys, "-o", color=_mech_color(m), label=m, lw=2, ms=6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("confabulation rate" + (" (n=50, 95% Wilson CI)" if hardened else ""))
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Each model verifies a DIFFERENT subset of identity dimensions (lines cross)\n"
                 "— yet all stop confabulating when no complete value can be COPIED (partial)")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_taxonomy.png", dpi=150)
    plt.close(fig)
    print("fig8_taxonomy.png" + (" (hardened+CI)" if hardened else ""))


def fig9_taxonomy_heatmap():
    """Model x dimension heatmap — scales to the full roster better than fig8's
    lines. Makes the 'attr column is uniformly hot (except Qwen3.5), partial column
    uniformly cold' pattern pop."""
    rows = _load("taxonomy_hardened.json") or _load("taxonomy.json")
    if not rows:
        return
    order = ["irrelevant", "attr_mismatch", "stale", "partial"]
    collabels = ["irrelevant\n(entity)", "attr-mismatch\n(attribute)",
                 "stale\n(recency)", "partial\n(completeness)"]
    by = {}
    for r in rows:
        by.setdefault(r["model"], {"size": r["size_b"]})[r["cond"]] = r["confab_rate"]
    models = sorted(by, key=lambda m: by[m]["size"])
    mat = [[by[m].get(c, float("nan")) for c in order] for m in models]
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(models) + 1.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(collabels, fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m} ({by[m]['size']:g}B)" for m in models], fontsize=8)
    for i, m in enumerate(models):
        for j, c in enumerate(order):
            v = by[m].get(c)
            if v is not None:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (v < 0.25 or v > 0.75) else "black")
    ax.set_title("Confabulation by mismatch dimension (red = mis-grounds)\n"
                 "attribute column hot for all but Qwen3.5; partial column cold for all")
    fig.colorbar(im, ax=ax, label="confab rate", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS / "fig9_taxonomy_heatmap.png", dpi=150)
    plt.close(fig)
    print("fig9_taxonomy_heatmap.png")


def fig10_verifier_blindspot():
    """Cross-model verifier catch rate by mismatch dimension — does adding a
    verifier fix the blind spot? Entity yes (~1.0), attribute/recency no."""
    rows = _load("ma_dimensions.json")
    if not rows:
        return
    from collections import defaultdict
    order = ["irrelevant", "attr_mismatch", "stale"]
    xlabels = ["irrelevant\n(entity)", "attr-mismatch\n(attribute)", "stale\n(recency)"]
    byc = defaultdict(list)
    for r in rows:
        if (r["worker"] != r["supervisor"] and r["catch_rate"] is not None
                and r["n_confab"] >= 3):
            byc[r["cond"]].append(r["catch_rate"])
    means = [sum(byc[c]) / len(byc[c]) if byc[c] else float("nan") for c in order]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(range(len(order)), means, color=["#2ca02c", "#d62728", "#ff7f0e"], alpha=0.85)
    for c_i, c in enumerate(order):
        for v in byc[c]:
            ax.plot(c_i + 0.0, v, "o", color="black", ms=3, alpha=0.35)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("cross-model verifier catch rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does a second-agent VERIFIER fix the blind spot?\n"
                 "Entity mismatch: caught (~1.0). Attribute/recency: verifier shares the blind spot")
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_verifier_blindspot.png", dpi=150)
    plt.close(fig)
    print("fig10_verifier_blindspot.png")


def fig11_prompt_robustness():
    """confab under 3 prompt wordings. The DIRECTION (attr > entity) is invariant;
    absolute rates shift with wording (honest)."""
    rows = _load("prompt_robustness.json")
    if not rows:
        return
    from collections import defaultdict
    variants = ["orig", "reword_B", "reword_C"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    by = defaultdict(dict)
    for r in rows:
        by[(r["model"], r["cond"])][r["variant"]] = r["confab_rate"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(models))
    w = 0.12
    for vi, v in enumerate(variants):
        irr = [by[(m, "irrelevant")].get(v, float("nan")) for m in models]
        att = [by[(m, "attr_mismatch")].get(v, float("nan")) for m in models]
        ax.bar([i - 0.2 + vi * w for i in x], irr, w, color="#2ca02c",
               alpha=0.45 + 0.2 * vi, label=f"entity ({v})" if vi == 0 else None)
        ax.bar([i + 0.2 + vi * w for i in x], att, w, color="#d62728",
               alpha=0.45 + 0.2 * vi, label=f"attribute ({v})" if vi == 0 else None)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("confab rate (3 prompt wordings each)")
    ax.set_title("Across 3 prompt rewordings: attribute > entity holds everywhere\n"
                 "(direction invariant; absolute rate is prompt-sensitive — green=entity, red=attribute)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGS / "fig11_prompt_robustness.png", dpi=150)
    plt.close(fig)
    print("fig11_prompt_robustness.png")


def fig12_realistic_replication():
    """Template vs realistic-prose carriers — does the effect survive natural docs?
    Paired bars per (model, condition); close pairs ⇒ not a template artifact."""
    rows = _load("realistic.json")
    if not rows:
        return
    conds = ["irrelevant", "attr_mismatch", "stale", "partial"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    by = {}
    for r in rows:
        by[(r["model"], r["cond"], r["style"])] = r["confab_rate"]
    fig, axes = plt.subplots(1, len(conds), figsize=(14, 4), sharey=True)
    for ax, cond in zip(axes, conds):
        x = range(len(models))
        t = [by.get((m, cond, "template"), float("nan")) for m in models]
        r = [by.get((m, cond, "realistic"), float("nan")) for m in models]
        ax.bar([i - 0.2 for i in x], t, 0.4, label="template", color="#7f7f7f")
        ax.bar([i + 0.2 for i in x], r, 0.4, label="realistic", color="#1f77b4")
        ax.set_title(cond, fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=6)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("confab rate")
    axes[0].legend(fontsize=7)
    fig.suptitle("Ecological validity: template vs realistic document prose "
                 "(attribute blind spot & partial≈0 persist on natural docs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "fig12_realistic_replication.png", dpi=150)
    plt.close(fig)
    print("fig12_realistic_replication.png")


# --- 2026-06-25 session: the mechanism->mitigation chain (record benchmark) ----

def _pool(rows, model, condition, metric="confab"):
    """n-weighted pooled rate over domains from baseline.json-style rows."""
    sel = [r for r in rows if r["model"] == model and r["condition"] == condition
           and r["metric"] == metric]
    n = sum(r["n"] for r in sel)
    return (sum(r["rate"] * r["n"] for r in sel) / n if n else float("nan")), n


def fig13_mitigation_ladder():
    """HEADLINE — which layer does mitigation live in? attr_mismatch mis-grounding
    drops only when we change the INPUT (absence/isolate), not when we prompt (cite).
    Pooled over models, with thin per-model lines behind the bars."""
    rows = _load("input_cure.json")
    if not rows:
        return
    levers = ["orig", "cite", "absence", "tags", "isolate"]
    labels = ["orig\n(do nothing)", "cite\n(PROMPT)", "absence\n(input:\nsalience)",
              "tags\n(input:\nsalience)", "isolate\n(input:\nSTRUCTURAL)"]
    colors = {"orig": "#7f7f7f", "cite": "#ff7f0e", "absence": "#1f77b4",
              "tags": "#1f77b4", "isolate": "#2ca02c"}
    pooled, by_model = {}, {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["lever"]] = r["confab"]
    for lv in levers:
        sel = [r for r in rows if r["lever"] == lv]
        n = sum(r["n"] for r in sel)
        pooled[lv] = sum(r["confab"] * r["n"] for r in sel) / n if n else float("nan")
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(levers))
    for m, d in sorted(by_model.items()):
        ax.plot(xs, [d.get(lv, float("nan")) for lv in levers], "-", color=_mech_color(m),
                lw=1, alpha=0.35, zorder=1)
    ax.bar(xs, [pooled[lv] for lv in levers], color=[colors[lv] for lv in levers],
           alpha=0.85, zorder=2, width=0.6)
    for x, lv in zip(xs, levers):
        ax.text(x, pooled[lv] + 0.02, f"{pooled[lv]:.2f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.axvspan(-0.5, 1.5, color="#ffeeee", alpha=0.4, zorder=0)
    ax.axvspan(1.5, 4.5, color="#eef6ff", alpha=0.5, zorder=0)
    ax.text(0.5, 0.92, "ask the model", ha="center", fontsize=8, color="#aa3333",
            transform=ax.get_xaxis_transform())
    ax.text(3.0, 0.92, "change what the tool returns", ha="center", fontsize=8,
            color="#3366aa", transform=ax.get_xaxis_transform())
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("attribute-mismatch mis-grounding rate (pooled)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Mitigation lives in the input/tool layer, not the prompt\n"
                 "(prompt 'cite' is partial and can backfire on label-insensitive models; "
                 "withholding the wrong value -> ~0)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig13_mitigation_ladder.png", dpi=150)
    plt.close(fig)
    print("fig13_mitigation_ladder.png")


def fig14_record_profile_heatmap():
    """Model x 4-dimension confab heatmap on the RECORD benchmark (baseline, n=25).
    The Tool-Trust Card: attribute warms/cools with capability, recency does not,
    completeness cold for all."""
    rows = _load("baseline.json")
    if not rows:
        return
    order = ["irrelevant", "attr_mismatch", "stale", "partial"]
    collabels = ["irrelevant\n(entity)", "attr-mismatch\n(attribute)",
                 "stale\n(recency)", "partial\n(completeness)"]
    models = sorted({r["model"] for r in rows},
                    key=lambda m: next(r["size_b"] for r in rows if r["model"] == m))
    size = {m: next(r["size_b"] for r in rows if r["model"] == m) for m in models}
    mat = [[_pool(rows, m, c)[0] for c in order] for m in models]
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(models) + 1.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(collabels, fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m} ({size[m]:g}B)" for m in models], fontsize=8)
    for i, _m in enumerate(models):
        for j, _c in enumerate(order):
            v = mat[i][j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (v < 0.25 or v > 0.75) else "black")
    ax.set_title("Tool-Trust Card: silent-mismatch confab by dimension (record, n=25)\n"
                 "attribute cools with scale, recency does NOT, completeness cold for all")
    fig.colorbar(im, ax=ax, label="confab rate", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS / "fig14_record_profile_heatmap.png", dpi=150)
    plt.close(fig)
    print("fig14_record_profile_heatmap.png")


def fig15_logprob_copyproof():
    """Logit-level copy evidence: lp(decoy) is high only because the value is on the
    page; remove it (isolate) and lp(decoy) collapses.
    Reads the raw jsonl (per-item lp_decoy), so guard on its presence."""
    raw = Path("results/raw/logprob_margin_raw.jsonl")
    if not raw.exists():
        return
    import collections
    lpd = collections.defaultdict(list)
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["rung"] in ("orig", "isolate"):
            lpd[(r["model"], r["rung"])].append(r["lp_decoy"])
    models = sorted({m for (m, _) in lpd},
                    key=lambda m: list(_MECH_COLORS).index(m) if m in _MECH_COLORS else 99)
    if not models:
        return
    mean = {k: sum(v) / len(v) for k, v in lpd.items()}
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(models))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [mean[(m, "orig")] for m in models], w,
           label="orig (decoy on page)", color="#d62728", alpha=0.85)
    ax.bar([i + w / 2 for i in x], [mean[(m, "isolate")] for m in models], w,
           label="isolate (decoy removed)", color="#2ca02c", alpha=0.85)
    for i, m in enumerate(models):
        drop = mean[(m, "isolate")] - mean[(m, "orig")]
        ax.annotate(f"{drop:+.1f}", (i, max(mean[(m, "orig")], mean[(m, "isolate")]) + 0.2),
                    ha="center", fontsize=8, color="#444")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean log-prob of the decoy value")
    ax.set_title("Logit-level evidence for copying: the decoy is high-prob only while on the page\n"
                 "remove it and lp(decoy) collapses (page-conditional copying)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig15_logprob_copyproof.png", dpi=150)
    plt.close(fig)
    print("fig15_logprob_copyproof.png")


def fig16_scale_dissociation():
    """Scale fixes the attribute dimension (rho~-0.8) but NOT recency (rho~0):
    two independent axes => you must measure each. From baseline.json."""
    rows = _load("baseline.json")
    if not rows:
        return
    import math
    models = sorted({r["model"] for r in rows},
                    key=lambda m: next(r["size_b"] for r in rows if r["model"] == m))
    size = {m: next(r["size_b"] for r in rows if r["model"] == m) for m in models}

    def series(cond):
        return [_pool(rows, m, cond)[0] for m in models]

    def rho(xs, ys):
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        dy = math.sqrt(sum((b - my) ** 2 for b in ys))
        return num / (dx * dy) if dx and dy else float("nan")

    sizes = [size[m] for m in models]
    attr, rec = series("attr_mismatch"), series("stale")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(sizes, attr, color="#1f77b4", s=70, zorder=3,
               label=f"attribute  (r={rho(sizes, attr):+.2f}: scale helps)")
    ax.scatter(sizes, rec, color="#d62728", marker="s", s=70, zorder=3,
               label=f"recency    (r={rho(sizes, rec):+.2f}: scale does NOT)")
    # simple least-squares trend lines
    for ys, col in ((attr, "#1f77b4"), (rec, "#d62728")):
        n = len(sizes); mx = sum(sizes) / n; my = sum(ys) / n
        b = (sum((a - mx) * (c - my) for a, c in zip(sizes, ys))
             / sum((a - mx) ** 2 for a in sizes))
        a0 = my - b * mx
        xs = [min(sizes), max(sizes)]
        ax.plot(xs, [a0 + b * v for v in xs], "--", color=col, lw=1.5, alpha=0.7)
    for m in models:
        ax.annotate(m, (size[m], _pool(rows, m, "attr_mismatch")[0]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("model size (B)")
    ax.set_ylabel("confab rate (silent mismatch)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Scale fixes ATTRIBUTE but not RECENCY — independent axes\n"
                 "(one scalar 'capability' can't summarize tool-trust)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig16_scale_dissociation.png", dpi=150)
    plt.close(fig)
    print("fig16_scale_dissociation.png")


def main():
    fig1_calibration()
    fig2_uptake_vs_size()
    fig3_explicit_vs_silent()
    fig4_catch_matrix()
    fig5_plausibility_gradient()
    fig6_verify_intervention()
    fig7_lexical_signal()
    fig8_taxonomy()
    fig9_taxonomy_heatmap()
    fig10_verifier_blindspot()
    fig11_prompt_robustness()
    fig12_realistic_replication()
    fig13_mitigation_ladder()
    fig14_record_profile_heatmap()
    fig15_logprob_copyproof()
    fig16_scale_dissociation()
    print(f"\nsaved to {FIGS}/")


if __name__ == "__main__":
    main()
