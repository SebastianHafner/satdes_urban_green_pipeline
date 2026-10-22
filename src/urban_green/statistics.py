"""Functions to produce annual urban green statistics.

Each statistic has a *_from_counts variant built from per-year pixel counts, so it can
be accumulated tile by tile (see urban_green.pipeline) instead of from one in-memory stack.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def green_counts(smoothed_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-year (n_valid, n_green) pixel counts of a (T, H, W) {0, 1, NaN} stack."""
    n_valid = np.isfinite(smoothed_stack).sum(axis=(1, 2)).astype(np.int64)
    n_green = (smoothed_stack == 1).sum(axis=(1, 2)).astype(np.int64)
    return n_valid, n_green


def change_counts(change_year: np.ndarray, years: List[int]) -> np.ndarray:
    """Number of pixels per year in years[1:] whose change-year raster equals that year."""
    return np.array([int(np.sum(change_year == year)) for year in years[1:]], dtype=np.int64)


def green_area_stats_from_counts(
    n_valid: np.ndarray, n_green: np.ndarray, years: List[int], pixel_area_m2: float = 100.0
) -> pd.DataFrame:
    """Per-year green area (km^2), % cover of the valid (masked) area, and year-on-year change."""
    rows = []
    prev_area = None
    for i, year in enumerate(years):
        area_km2 = int(n_green[i]) * pixel_area_m2 / 1e6
        pct_cover = (n_green[i] / n_valid[i] * 100) if n_valid[i] else np.nan
        change_km2 = 0.0 if prev_area is None else area_km2 - prev_area
        rows.append(
            {"year": year, "green_area_km2": area_km2, "pct_cover": pct_cover, "change_km2": change_km2}
        )
        prev_area = area_km2
    return pd.DataFrame(rows)


def green_area_stats(smoothed_stack: np.ndarray, years: List[int], pixel_area_m2: float = 100.0) -> pd.DataFrame:
    """Per-year green area (km^2), % cover of the valid (masked) area, and year-on-year change."""
    return green_area_stats_from_counts(*green_counts(smoothed_stack), years, pixel_area_m2)


def green_area_stats_by_region_from_counts(
    region_counts: Dict[str, Tuple[np.ndarray, np.ndarray]], years: List[int], pixel_area_m2: float = 100.0
) -> pd.DataFrame:
    """green_area_stats_from_counts per named sub-region; region_counts maps name -> (n_valid, n_green)."""
    frames = []
    for name, (n_valid, n_green) in region_counts.items():
        df = green_area_stats_from_counts(n_valid, n_green, years, pixel_area_m2)
        df.insert(0, "region", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def green_area_stats_by_region(
    smoothed_stack: np.ndarray, years: List[int], region_masks: Dict[str, np.ndarray], pixel_area_m2: float = 100.0
) -> pd.DataFrame:
    """Same as green_area_stats, broken out per named sub-region (see urban_green.roi.region_masks)."""
    region_counts = {
        name: green_counts(np.where(mask[None, :, :], smoothed_stack, np.nan)) for name, mask in region_masks.items()
    }
    return green_area_stats_by_region_from_counts(region_counts, years, pixel_area_m2)


def change_summary_from_counts(
    loss_counts: np.ndarray, gain_counts: np.ndarray, years: List[int], pixel_area_m2: float = 100.0
) -> pd.DataFrame:
    """Per-year green loss/gain area (km^2); counts are per year in years[1:] (see change_counts)."""
    rows = []
    for i, year in enumerate(years[1:]):
        loss_km2 = int(loss_counts[i]) * pixel_area_m2 / 1e6
        gain_km2 = int(gain_counts[i]) * pixel_area_m2 / 1e6
        rows.append({"year": year, "loss_km2": loss_km2, "gain_km2": gain_km2})
    return pd.DataFrame(rows)


def change_summary(
    loss_year: np.ndarray, gain_year: np.ndarray, years: List[int], pixel_area_m2: float = 100.0
) -> pd.DataFrame:
    """Per-year green loss/gain area (km^2) derived from the change-year rasters."""
    return change_summary_from_counts(
        change_counts(loss_year, years), change_counts(gain_year, years), years, pixel_area_m2
    )
