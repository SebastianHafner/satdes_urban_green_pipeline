"""Green loss/gain change-year products.

Each output raster encodes, per pixel, the year of the FIRST transition in
one direction (vegetated->non-vegetated for loss, non-vegetated->vegetated
for gain) across the smoothed annual classification stack. Pixels with no
such transition keep the fill value (default 0, which never collides with a
real calendar year).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio


def compute_change_years(veg_stack: np.ndarray, years: List[int], fill_value: int = 0) -> Dict[str, np.ndarray]:
    """veg_stack: (T, H, W) with values in {0, 1, NaN} (NaN = outside ROI/mask).

    Returns {"loss_year": (H, W) uint16, "gain_year": (H, W) uint16}.
    """
    n_years, h, w = veg_stack.shape
    if len(years) != n_years:
        raise ValueError("years must match the length of veg_stack's first axis.")

    loss_year = np.full((h, w), fill_value, dtype=np.uint16)
    gain_year = np.full((h, w), fill_value, dtype=np.uint16)
    loss_found = np.zeros((h, w), dtype=bool)
    gain_found = np.zeros((h, w), dtype=bool)

    for t in range(1, n_years):
        prev, curr = veg_stack[t - 1], veg_stack[t]
        valid = np.isfinite(prev) & np.isfinite(curr)

        loss = valid & (prev == 1) & (curr == 0) & ~loss_found
        gain = valid & (prev == 0) & (curr == 1) & ~gain_found

        loss_year[loss] = years[t]
        gain_year[gain] = years[t]
        loss_found |= loss
        gain_found |= gain

    return {"loss_year": loss_year, "gain_year": gain_year}


def write_change_geotiff(array: np.ndarray, transform, crs, out_file: Path, fill_value: int = 0) -> Path:
    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint16",
        crs=crs,
        transform=transform,
        nodata=fill_value,
    ) as dst:
        dst.write(array, 1)
    return out_file
