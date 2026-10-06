#!/usr/bin/env python3
"""Figures for the README, drawn from results/folktables_2018.csv only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"

TASKS = {  # fixed order: color follows the task, marker is a second encoding
    "income": ("Income", "#2a78d6", "o"),
    "employment": ("Employment", "#eb6834", "s"),
    "public_coverage": ("Public coverage", "#1baf7a", "^"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(ROOT / "results" / "folktables_2018.csv")
    FIG.mkdir(exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, dpi=150)

    for key, (label, color, marker) in TASKS.items():
        g = df[df["task"] == key]
        kw = dict(s=34, color=color, marker=marker, edgecolor="white", linewidth=0.8, label=label)
        a1.scatter(g["psi_max"], 100 * g["acc_drop"], **kw)
        a2.scatter(100 * g["atc_drop"], 100 * g["acc_drop"], **kw)

    for ax in (a1, a2):
        style(ax)
        ax.axhline(2, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    a1.text(0.07, 2.4, "degraded (> 2 pt drop)", color=MUTED, fontsize=8)

    a1.set_xscale("log")
    a1.axvline(0.25, color=INK, linewidth=1)
    a1.text(0.27, -13.5, "PSI alarm\nthreshold 0.25", color=INK, fontsize=8)
    a1.set_xlabel("Input drift: max per-feature PSI (log scale)", color=INK, fontsize=10)
    a1.set_ylabel("Actual accuracy drop vs CA holdout (points)", color=INK, fontsize=10)
    a1.set_title("Drift alarms on almost every state", color=INK, fontsize=11, loc="left")

    lim = np.array([-15, 11])
    a2.plot(lim, lim, color=MUTED, linewidth=1)
    a2.text(6.2, 8.6, "y = x", color=MUTED, fontsize=8)
    a2.set_xlabel("ATC-estimated accuracy drop, no labels (points)", color=INK, fontsize=10)
    a2.set_title("ATC tracks the real drop", color=INK, fontsize=11, loc="left")
    a2.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(
        "Train on California, deploy to 49 states x 3 ACS tasks (2018, 147 deployments)",
        color=MUTED,
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "drift_vs_atc.png", facecolor="white")
    print("wrote", FIG / "drift_vs_atc.png")


if __name__ == "__main__":
    main()
