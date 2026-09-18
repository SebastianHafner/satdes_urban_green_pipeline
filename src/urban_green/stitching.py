"""Mosaic per-tile annual max-NDVI rasters into one raster per year.

Adapted from satdes_maxndvi/stitching.py: generalized to work on the
float16 output of urban_green.acquisition and to group tiles by year
rather than by a fixed tiled-file naming scheme. Unlike the original,
source tiles are kept on disk (not deleted) after stitching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.merge import merge


def stitch_year(tile_files: List[Path], out_file: Path, dtype: str = "float16", nodata=np.nan) -> Path:
    srcs = [rasterio.open(f) for f in tile_files]
    try:
        mosaic, transform = merge(srcs, nodata=nodata)
        profile = srcs[0].profile.copy()
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, dtype=dtype, nodata=nodata)
        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(mosaic.astype(dtype))
    finally:
        for src in srcs:
            src.close()
    return out_file


def stitch_acquisition(acquisition_dir: Path, years: List[int], out_dir: Path) -> Dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stitched = {}
    for year in years:
        tile_dir = acquisition_dir / f"y{year}" / "tiles"
        ndvi_tiles = sorted(tile_dir.glob("ndvi_*.tif"))
        if not ndvi_tiles:
            raise FileNotFoundError(f"No NDVI tiles found for year {year} in {tile_dir}")
        out_file = out_dir / f"maxndvi_{year}.tif"
        stitch_year(ndvi_tiles, out_file)
        stitched[year] = out_file
    return stitched
