"""Annual urban green statistics, akin to the Overleaf thresholding report's results tables."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def green_area_stats(smoothed_stack: np.ndarray, years: List[int], pixel_area_m2: float = 100.0) -> pd.DataFrame:
    """Per-year green area (km^2), % cover of the valid (masked) area, and year-on-year change."""
    rows = []
    prev_area = None
    for i, year in enumerate(years):
        layer = smoothed_stack[i]
        n_valid = int(np.sum(np.isfinite(layer)))
        n_green = int(np.nansum(layer == 1))
        area_km2 = n_green * pixel_area_m2 / 1e6
        pct_cover = (n_green / n_valid * 100) if n_valid else np.nan
        change_km2 = 0.0 if prev_area is None else area_km2 - prev_area
        rows.append(
            {"year": year, "green_area_km2": area_km2, "pct_cover": pct_cover, "change_km2": change_km2}
        )
        prev_area = area_km2
    return pd.DataFrame(rows)


def green_area_stats_by_region(
    smoothed_stack: np.ndarray, years: List[int], region_masks: Dict[str, np.ndarray], pixel_area_m2: float = 100.0
) -> pd.DataFrame:
    """Same as green_area_stats, broken out per named sub-region (see urban_green.roi.region_masks)."""
    frames = []
    for name, mask in region_masks.items():
        masked_stack = np.where(mask[None, :, :], smoothed_stack, np.nan)
        df = green_area_stats(masked_stack, years, pixel_area_m2)
        df.insert(0, "region", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def change_summary(
    loss_year: np.ndarray, gain_year: np.ndarray, years: List[int], pixel_area_m2: float = 100.0
) -> pd.DataFrame:
    """Per-year green loss/gain area (km^2) derived from the change-year rasters."""
    rows = []
    for year in years[1:]:
        loss_km2 = int(np.sum(loss_year == year)) * pixel_area_m2 / 1e6
        gain_km2 = int(np.sum(gain_year == year)) * pixel_area_m2 / 1e6
        rows.append({"year": year, "loss_km2": loss_km2, "gain_km2": gain_km2})
    return pd.DataFrame(rows)
