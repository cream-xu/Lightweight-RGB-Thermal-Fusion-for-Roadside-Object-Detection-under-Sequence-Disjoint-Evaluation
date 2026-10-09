"""Reproduce manuscript Figs. 4 and 6 from recorded results."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"
FILLED = True

# Reference family: compact, white, three categorical hues and neutral controls.
BLUE = "#0072B2"       # Concat / four-channel reference
GREEN = "#009E73"      # GSW / added-depth alternative
PINK = "#CC79A7"       # GBF / late fusion
GRAY = "#85898D"       # TBF or neutral
LIGHT = "#D7DEE2"
GRID = "#E3E7EA"
TEXT = "#20272D"
ORANGE = "#DE8F05"     # Sparse numeric emphasis only
MODEL_COLOR = {"Concat": BLUE, "TBF": GRAY, "GSW": GREEN, "GBF": PINK}
MODEL_MARKER = {"Concat": "o", "TBF": "s", "GSW": "^", "GBF": "D"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.1, "axes.labelsize": 8.1,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": "#59636A", "axes.linewidth": .75,
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "svg.fonttype": "none", "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def read(path):
    mapping = {"rlivit_msq/summary.json": "msq_summary.json",
               "llvip_v2/summary.json": "llvip_summary.json"}
    filename = mapping.get(path, Path(path).name)
    return json.loads((RESULTS / filename).read_text(encoding="utf-8"))


def save(fig, name):
    filename = {"fig4_reference_style": "fig4_illumination_bootstrap",
                "fig6_reference_style": "fig6_depth_llvip"}[name]
    fig.savefig(OUT / (filename + ".png"), dpi=400)
    fig.savefig(OUT / (filename + ".svg"))
    plt.close(fig)


def eval_record(model, seed):
    stem = {"Concat": "sq-4ch", "TBF": "sq-4ch-tbf",
            "GSW": "sq-4ch-gsw", "GBF": "sq-4ch-gbf"}[model]
    return read(f"rlivit_sq/eval_{stem}_s{seed}.json")


def val(model, seed, scope="full", metric="AP50"):
    return 100 * eval_record(model, seed)[scope][metric]


def panel(ax, letter, title, note=None):
    ax.text(-.04, 1.10, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=11, va="bottom", ha="right")
    ax.set_title(title, loc="left", fontsize=8.7, pad=13 if note else 8)
    if note:
        ax.text(0, 1.025, note, transform=ax.transAxes, fontsize=7.1,
                color="#69737A", va="bottom")


def axes_style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=.6, pad=3)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.55)
        ax.set_axisbelow(True)


def footer(fig, text):
    fig.text(.06, .022, text, fontsize=7.0, color="#69737A")


def model_legend(ax, labels, *, loc="upper right", ncol=None, bbox=None):
    handles = [Line2D([], [], marker=MODEL_MARKER[m], color=MODEL_COLOR[m],
                      linestyle="none", markersize=5, label=m) for m in labels]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, frameon=False,
              ncol=ncol or len(labels), fontsize=7.0, handletextpad=.3,
              columnspacing=.7, borderpad=.15)


def grouped_seed_panel(ax, scope, letter, title, ylimits):
    models = ["Concat", "TBF", "GSW", "GBF"]
    for seed in range(3):
        present = models if seed == 0 else ["Concat", "GSW", "GBF"]
        vals = [val(m, seed, scope=scope) for m in present]
        base = seed
        if FILLED:
            ax.add_patch(Rectangle((base-.40, min(vals)), .80,
                                   max(vals)-min(vals), facecolor=GRAY,
                                   alpha=.10, edgecolor="none", zorder=1))
        width = .18
        offsets = {"Concat": -.29, "TBF": -.095, "GSW": .095, "GBF": .29}
        for m, v in zip(present, vals):
            ax.bar(base+offsets[m], v, width=width, color=MODEL_COLOR[m],
                   edgecolor="white", lw=.4, alpha=.91, zorder=2)
    ax.set_xticks([0, 1, 2], ["Seed 0", "Seed 1", "Seed 2"])
    ax.set_ylim(*ylimits)
    ax.set_ylabel("mAP50 (%)")
    panel(ax, letter, title, "TBF seed 0 only; pale band = model span" if FILLED else "TBF has seed 0 only")
    axes_style(ax)
    model_legend(ax, models, loc="upper right", ncol=2)


def figure4():
    fig = plt.figure(figsize=(8.15, 6.15))
    gs = fig.add_gridspec(2, 2, left=.115, right=.965, top=.90, bottom=.105,
                          wspace=.39, hspace=.63)
    a, b, c, d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    grouped_seed_panel(a, "full", "a", "Full-test accuracy", (0, 70))
    grouped_seed_panel(b, "night", "b", "Nighttime accuracy", (0, 62))

    # Numeric signed matrix is more legible than 21 tiny bars at manuscript width.
    classes = list(eval_record("Concat", 0)["per_class_ap50"])
    methods = ["TBF", "GSW", "GBF"]
    signed = np.array([[(eval_record(m, 0)["per_class_ap50"][cl] -
                         eval_record("Concat", 0)["per_class_ap50"][cl]) * 100
                        for m in methods] for cl in classes])
    for i, cl in enumerate(classes):
        for j, m in enumerate(methods):
            v = signed[i, j]
            color = GREEN if v > .05 else PINK if v < -.05 else GRAY
            c.scatter(j, len(classes)-1-i, s=405, marker="s", color=color,
                      alpha=.17 + .45 * min(abs(v)/6, 1), edgecolors="none")
            c.text(j, len(classes)-1-i, f"{v:+.1f}", ha="center", va="center",
                   fontsize=7.5, color=TEXT, fontweight="bold" if abs(v) >= 2 else "normal")
    c.set_xticks(range(3), methods)
    c.set_yticks(range(len(classes)), classes[::-1])
    c.set_xlim(-.5, 2.5); c.set_ylim(-.5, len(classes)-.5)
    c.tick_params(length=0)
    for spine in c.spines.values(): spine.set_visible(False)
    panel(c, "c", "Class-wise change vs Concat", "Seed 0 · Δ mAP50 (percentage points)")

    rows = [("GBF − Concat, full", "concat_vs_gbf_full", PINK),
            ("GBF − Concat, night", "concat_vs_gbf_night", PINK),
            ("GSW − Concat, full", "concat_vs_gsw_full", GREEN),
            ("5ch − 4ch, depth", "depth_4ch_vs_5ch", GRAY)]
    boot = read("rlivit_sq/bootstrap_seq.json")["comparisons"]
    for yi, (label, key, color) in enumerate(rows[::-1]):
        record = boot[key]
        lo, hi = [100*x for x in record["dAP50_ci95"]]
        mid = 100*record["dAP50_point"]
        if FILLED:
            d.add_patch(Rectangle((lo, yi-.20), hi-lo, .40,
                                  facecolor=color, alpha=.21,
                                  edgecolor="none", zorder=1))
        d.plot([lo, hi], [yi, yi], color=color, lw=2.5, solid_capstyle="butt")
        d.plot([lo, lo], [yi-.10, yi+.10], color=color, lw=.8)
        d.plot([hi, hi], [yi-.10, yi+.10], color=color, lw=.8)
        d.scatter(mid, yi, s=32, color=color, edgecolor="white", zorder=3)
    d.axvline(0, color=ORANGE, linestyle="--", lw=1)
    d.set_yticks(range(4), [r[0] for r in rows[::-1]])
    bounds = [100*x for record in boot.values() for x in record["dAP50_ci95"]]
    d.set_xlim(min(-4, min(bounds)-.5), max(10, max(bounds)+.5)); d.set_xticks([-4, 0, 4, 8])
    d.set_xlabel("Δ mAP50 (pp)")
    panel(d, "d", "Sequence-bootstrap estimates", "Original-test difference and 95% percentile CI")
    axes_style(d, "x")
    footer(fig, "Paired sequence bootstrap retains repeated draws; dots are original-test differences. Training-seed uncertainty is excluded.")
    save(fig, "fig4_reference_style")


def figure6():
    fig = plt.figure(figsize=(8.15, 5.95))
    gs = fig.add_gridspec(2, 2, left=.12, right=.965, top=.90, bottom=.11,
                          wspace=.40, hspace=.61)
    a, b, c, d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    msq = read("rlivit_msq/summary.json")
    for seed in range(3):
        v4 = 100*msq[f"msq-4ch_s{seed}"]["mAP50"]
        v5 = 100*msq[f"msq-5ch_s{seed}"]["mAP50"]
        y = 2-seed
        if FILLED:
            fill = GREEN if v5 >= v4 else PINK
            a.barh(y, abs(v5-v4), left=min(v4, v5), height=.25,
                   color=fill, alpha=.32, edgecolor="none", zorder=1)
            a.plot([v4, v5], [y, y], color=fill, lw=1.0, zorder=2)
        else:
            a.plot([v4, v5], [y, y], color="#B9C3C9", lw=2)
        a.scatter(v4, y, color=BLUE, s=40, marker="o", edgecolor="white", zorder=3)
        a.scatter(v5, y, color=GREEN, s=40, marker="s", edgecolor="white", zorder=3)
        a.text(max(v4, v5)+.15, y, f"{v5-v4:+.2f}", va="center",
               color=GREEN if v5 >= v4 else BLUE, fontsize=7.4)
    a.set_yticks([0, 1, 2], ["Seed 2", "Seed 1", "Seed 0"])
    a.set_xlim(37, 41); a.set_xticks([37, 38, 39, 40, 41])
    a.set_xlabel("mAP50 (%)")
    panel(a, "a", "Projected-depth response", "Blue = 4ch; green = 5ch · matched seeds")
    axes_style(a, "x")

    configs = [("4ch Concat", "msq-4ch_s0", BLUE),
               ("5ch Concat", "msq-5ch_s0", GREEN),
               ("4ch GBF", "msq-4ch-gbf_s0", BLUE),
               ("5ch GBF", "msq-5ch-gbf_s0", GREEN),
               ("5ch TBF", "msq-5ch-tbf_s0", GREEN)]
    for yi, (name, key, color) in enumerate(configs[::-1]):
        score = 100*msq[key]["mAP50"]
        if FILLED:
            b.barh(yi, score, color=color, height=.77, alpha=.20,
                   edgecolor="none", zorder=1)
        b.barh(yi, score, color=color, height=.48 if FILLED else .58,
               edgecolor="white", zorder=2)
        b.text(score+.18, yi, f"{score:.2f}", fontsize=7.5, va="center", color=color)
    b.set_yticks(range(5), [x[0] for x in configs[::-1]])
    b.set_xlim(0, 44); b.set_xticks([0, 10, 20, 30, 40]); b.set_xlabel("mAP50 (%)")
    panel(b, "b", "Depth configurations", "msq protocol; seed 0 only")
    axes_style(b, "x")

    ll = read("llvip_v2/summary.json")
    ll_rows = []
    for metric, name in [("test_mAP50", "mAP50"), ("test_mAP50-95", "mAP50–95")]:
        for seed in range(2):
            c0 = 100*ll[f"llvip-concat_s{seed}"][metric]
            gbf = 100*ll[f"llvip-gbf_s{seed}"][metric]
            ll_rows.append((name, seed, gbf-c0))
    for yi, (metric, seed, delta) in enumerate(ll_rows[::-1]):
        color = GREEN if delta > 0 else PINK
        if FILLED:
            c.barh(yi, abs(delta), left=min(0, delta), height=.31,
                   color=color, alpha=.26, edgecolor="none", zorder=1)
        c.plot([0, delta], [yi, yi], color=color, lw=2)
        c.scatter(delta, yi, marker="D", color=color, s=36, edgecolor="white", zorder=3)
        c.text(delta+(.09 if delta >= 0 else -.09), yi, f"{delta:+.2f}",
               ha="left" if delta >= 0 else "right", va="center", fontsize=7.3, color=color)
    c.axvline(0, color=ORANGE, linestyle="--", lw=.9)
    c.set_yticks(range(4), [f"{m} · s{s}" for m, s, _ in ll_rows[::-1]])
    c.set_xlim(-.75, 2.05); c.set_xticks([-.5, 0, .5, 1, 1.5, 2])
    c.set_xlabel("GBF − Concat (pp)")
    panel(c, "c", "LLVIP: metric-dependent change", "Separate retraining; 3,463 test pairs")
    axes_style(c, "x")

    e = read("rlivit_sq/bootstrap_seq.json")["comparisons"]["depth_4ch_vs_5ch"]
    lo, hi = [100*x for x in e["dAP50_ci95"]]
    mid = 100*e["dAP50_point"]
    d.axvline(0, color=ORANGE, linestyle="--", lw=.9)
    if FILLED:
        d.add_patch(Rectangle((lo, .31), hi-lo, .38,
                              facecolor=GREEN, alpha=.21,
                              edgecolor="none", zorder=1))
    d.plot([lo, hi], [.5, .5], color=GREEN, lw=3)
    d.plot([lo, lo], [.39, .61], color=GREEN, lw=1)
    d.plot([hi, hi], [.39, .61], color=GREEN, lw=1)
    d.scatter(mid, .5, color=GREEN, s=65, edgecolor="white", zorder=3)
    d.text(mid, .69, f"{mid:+.2f} pp", ha="center", color=GREEN,
           fontsize=8.5, fontweight="bold")
    d.text(.5, .17, f"95% CI [{lo:+.2f}, {hi:+.2f}] pp", transform=d.transAxes,
           ha="center", fontsize=7.2, color="#69737A")
    d.set_ylim(0, 1); d.set_yticks([])
    d.set_xlim(min(-4, lo-.5), max(8, hi+.5)); d.set_xticks([-4, -2, 0, 2, 4, 6, 8])
    d.set_xlabel("Δ mAP50: 5ch − 4ch (pp)")
    panel(d, "d", "Depth sequence bootstrap", "Saved seed-0 predictions; COCO evaluation")
    axes_style(d, "x")
    footer(fig, "Panels a-c: recorded training-framework AP. Panel d: COCO bootstrap with sequence multiplicity retained.")
    save(fig, "fig6_reference_style")

if __name__ == "__main__":
    figure4()
    figure6()
