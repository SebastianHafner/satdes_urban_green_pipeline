"""Mosaic per-tile annual max-NDVI rasters into one raster per year.

Adapted from satdes_maxndvi/stitching.py: generalized to work on the
float16 output of urban_green.acquisition, and to resolve tiles through
the shared urban_green.tile_store.TileStore (rather than globbing a
per-run directory), so only the tiles relevant to the current ROI are
mosaicked even when the store holds many more tiles overall.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine

from urban_green.grid import Tile
from urban_green.tile_store import TileStore


def mosaic_grid(path: Path) -> Tuple[Affine, Tuple[int, int]]:
    """Read the (transform, (height, width)) of an already-stitched mosaic.

    Since tiles come from the fixed national grid, every year's mosaic for
    the same ROI is expected to share this grid exactly -- see
    urban_green.pipeline._build_ndvi_stack, which uses this as the common
    working grid instead of resampling each year onto a separately
    computed one.
    """
    with rasterio.open(path) as src:
        return src.transform, (src.height, src.width)


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


def stitch_from_store(
    tile_store: TileStore, tiles: List[Tile], years: List[int], out_dir: Path
) -> Dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stitched = {}
    for year in years:
        ndvi_tiles = []
        for tile in tiles:
            record = tile_store.is_available(tile.row, tile.col, year)
            if record is None:
                raise FileNotFoundError(
                    f"Tile ({tile.row}, {tile.col}) for year {year} is not in the tile store "
                    f"at {tile_store.root} -- run acquisition before stitching."
                )
            ndvi_tiles.append(record.ndvi_path)
        out_file = out_dir / f"maxndvi_{year}.tif"
        stitch_year(ndvi_tiles, out_file)
        stitched[year] = out_file
    return stitched
