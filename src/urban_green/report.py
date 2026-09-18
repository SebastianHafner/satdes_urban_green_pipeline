"""Figures in the same style as the Overleaf thresholding report, for direct comparability."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_ndvi_distribution(values: np.ndarray, threshold: float, out_file: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    kde = gaussian_kde(values)
    x = np.linspace(-0.5, 1, 500)
    ax.plot(x, kde(x), color="forestgreen", label="KDE")
    ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
    ax.fill_between(x, kde(x), where=(x >= threshold), color="forestgreen", alpha=0.3, label="Green area")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_green_area_timeseries(stats: pd.DataFrame, threshold: float, out_file: Path) -> Path:
    years = stats["year"].tolist()
    areas_km2 = stats["green_area_km2"].tolist()
    change_values = stats["change_km2"].tolist()

    fig, (ax1, ax_bar) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(years, areas_km2, marker="o", color="forestgreen", linewidth=2, markersize=8)
    ax1.fill_between(years, areas_km2, color="forestgreen", alpha=0.2)
    ax1.set_ylabel("Green area (km²)", color="forestgreen")
    ax1.tick_params(axis="y", labelcolor="forestgreen")

    ax2 = ax1.twinx()
    ax2.axhline(threshold, color="red", linewidth=1.8, linestyle="--", label="Threshold")
    ax2.set_ylabel("Threshold (NDVI)", color="red")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="red")

    colors = ["forestgreen" if c >= 0 else "tomato" for c in change_values]
    ax_bar.bar(years, change_values, color=colors, edgecolor="white", width=0.6)
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.set_ylabel("Change in green area (km²)")
    ax_bar.set_xlabel("Year")

    plt.xticks(years)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file
