"""Regenerate the three paper figures at true column width (R9, 2026-08).

The earlier versions were drawn at 7.5-8in wide and then scaled to \\columnwidth
(3.03in) in LaTeX, so every label rendered at ~40% of its nominal size; fig13's
in-figure title was also clipped by the canvas and fig16 still used the retired
term "confab". These are drawn at 3.03in so 7pt is 7pt on the page, use an
Okabe-Ito colourblind-safe palette, carry no in-figure titles (the caption makes
the claim), and use the paper's vocabulary (mis-grounding).

    python debug/figures_paper.py        # writes paper/figs/ and results/figs/
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.03            # \columnwidth in inches (acl.sty, 11pt letter)
OUT = [Path("paper/figs"), Path("results/figs")]
DPI = 400

# Okabe-Ito: safe under deuteranopia/protanopia/tritanopia
BLUE, VERM, ORANGE, GREY = "#0072B2", "#D55E00", "#E69F00", "#8C8C8C"
SKY = "#56B4E9"

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

ROSTER = ["Phi-4-mini", "aya-expanse-8b", "gemma-3-12b", "Phi-4",
          "Mistral-Small-24B", "gemma-4-26b", "Qwen3.5-27B", "Qwen3.6-35B"]


def _load(name):
    p = Path("results") / name
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name):
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name)
    plt.close(fig)
    print(name)


def fig13_mitigation_ladder():
    """Attribute mis-grounding by rung: pooled bars, thin per-model lines."""
    rows = _load("input_cure.json")
    levers = ["orig", "cite", "absence", "tags", "isolate"]
    labels = ["orig", "cite", "absence", "tags", "isolate"]
    colors = [GREY, ORANGE, SKY, SKY, BLUE]
    by_model = collections.defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["lever"]] = r["confab"]
    pooled = {}
    for lv in levers:
        sel = [r for r in rows if r["lever"] == lv]
        n = sum(r["n"] for r in sel)
        pooled[lv] = sum(r["confab"] * r["n"] for r in sel) / n

    # compact: this figure sits in the body, where vertical space is the binding
    # constraint (ARR 8-page content limit).
    fig, ax = plt.subplots(figsize=(COL_W, 1.28))
    xs = list(range(len(levers)))
    ax.axvspan(-0.5, 1.5, color="#F2F2F2", zorder=0)
    ax.bar(xs, [pooled[lv] for lv in levers], color=colors, width=0.62,
           zorder=2, edgecolor="none")
    for m in ROSTER:
        d = by_model.get(m, {})
        ax.plot(xs, [d.get(lv, float("nan")) for lv in levers], "-",
                color="#444444", lw=0.5, alpha=0.35, zorder=3)
    for x, lv in zip(xs, levers):
        ax.text(x, pooled[lv] + 0.03, f"{pooled[lv]:.2f}", ha="center",
                fontsize=6.5, fontweight="bold", zorder=4)
    ax.text(0.5, 0.93, "ask the model", ha="center", fontsize=6, color="#555555",
            style="italic", transform=ax.get_xaxis_transform())
    ax.text(3.0, 0.93, "change what the tool returns", ha="center", fontsize=6,
            color="#555555", style="italic", transform=ax.get_xaxis_transform())
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("mis-grounding")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    _save(fig, "fig13_mitigation_ladder.png")


def fig15_logprob_copyproof():
    """log p(decoy) with the decoy on the page vs. removed, per model."""
    raw = Path("results/raw/logprob_margin_raw.jsonl")
    lpd = collections.defaultdict(list)
    for line in raw.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["rung"] in ("orig", "isolate"):
                lpd[(r["model"], r["rung"])].append(r["lp_decoy"])
    models = [m for m in ROSTER if (m, "orig") in lpd]
    mean = {k: sum(v) / len(v) for k, v in lpd.items()}

    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    x = list(range(len(models)))
    w = 0.36
    ax.bar([i - w / 2 for i in x], [mean[(m, "orig")] for m in models], w,
           label="decoy on the page", color=VERM, edgecolor="none")
    ax.bar([i + w / 2 for i in x], [mean[(m, "isolate")] for m in models], w,
           label="decoy removed", color=BLUE, edgecolor="none")
    for i, m in enumerate(models):
        drop = mean[(m, "isolate")] - mean[(m, "orig")]
        ax.annotate(f"{drop:.1f}", (i, 0.15), ha="center", fontsize=6,
                    color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("mean $\\log p$(decoy)")
    ax.set_ylim(-10.6, 0.9)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=1.1,
              borderpad=0.2, labelspacing=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    _save(fig, "fig15_logprob_copyproof.png")


def fig16_scale_dissociation():
    """Mis-grounding vs. model size: attribute tracks scale, recency does not."""
    rows = _load("baseline.json")
    size = {m: next(r["size_b"] for r in rows if r["model"] == m) for m in ROSTER}

    def pooled(m, cond):
        sel = [r for r in rows if r["model"] == m and r["condition"] == cond
               and r["metric"] == "confab"]
        n = sum(r["n"] for r in sel)
        return sum(r["rate"] * r["n"] for r in sel) / n

    sizes = [size[m] for m in ROSTER]
    attr = [pooled(m, "attr_mismatch") for m in ROSTER]
    rec = [pooled(m, "stale") for m in ROSTER]

    def rho(xs, ys):
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        dy = math.sqrt(sum((b - my) ** 2 for b in ys))
        return num / (dx * dy)

    def rlabel(name, ys):
        v = f"{rho(sizes, ys):.2f}".replace("-", "{-}")
        return f"{name} ($r{{=}}{v}$)"

    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    ax.scatter(sizes, attr, color=BLUE, s=22, zorder=3,
               label=rlabel("attribute", attr))
    ax.scatter(sizes, rec, color=VERM, marker="s", s=22, zorder=3,
               label=rlabel("recency", rec))
    for ys, col in ((attr, BLUE), (rec, VERM)):
        n = len(sizes); mx = sum(sizes) / n; my = sum(ys) / n
        b = (sum((a - mx) * (c - my) for a, c in zip(sizes, ys))
             / sum((a - mx) ** 2 for a in sizes))
        a0 = my - b * mx
        xs = [min(sizes) - 1, max(sizes) + 1]
        ax.plot(xs, [a0 + b * v for v in xs], "--", color=col, lw=0.9, alpha=0.8,
                zorder=2)
    # the within-model dissociation, made concrete on one model
    i = ROSTER.index("gemma-4-26b")
    ax.annotate("", xy=(sizes[i], rec[i] - 0.02), xytext=(sizes[i], attr[i] + 0.02),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="#555555"), zorder=4)
    ax.annotate("gemma-4-26b", (sizes[i], (attr[i] + rec[i]) / 2), fontsize=6,
                color="#333333", ha="left", va="center", rotation=90,
                xytext=(3, 0), textcoords="offset points",
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlabel("model size (B total parameters)")
    ax.set_ylabel("mis-grounding rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0, 39)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower left", frameon=False, handlelength=1.0, borderpad=0.2,
              labelspacing=0.25, handletextpad=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    _save(fig, "fig16_scale_dissociation.png")


if __name__ == "__main__":
    fig13_mitigation_ladder()
    fig15_logprob_copyproof()
    fig16_scale_dissociation()
