"""Region-of-interest ingestion: load, tile, and rasterize an arbitrary Swedish ROI.

Tiling resolves ROI features against the Landmateriet grid, so tile
identity (row, col) is independent of the requesting ROI and reusable
across runs via the tile store.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin

from urban_green.grid import GRID_CRS, TILE_SIZE_M_DEFAULT, Tile, national_grid_tiles


def load_roi(roi_path: Path, crs: str) -> gpd.GeoDataFrame:
    """Load a user-supplied ROI vector file and reproject it to the working CRS."""
    roi_path = Path(roi_path)
    gdf = gpd.read_parquet(roi_path) if roi_path.suffix == ".parquet" else gpd.read_file(roi_path)
    if gdf.crs is None:
        raise ValueError(f"ROI file {roi_path} has no CRS defined; cannot reproject safely.")
    if len(gdf) == 0:
        raise ValueError(f"ROI file {roi_path} contains no features.")
    return gdf.to_crs(crs)


def tile_roi(roi: gpd.GeoDataFrame, tile_size: float = TILE_SIZE_M_DEFAULT) -> List[Tile]:
    """Resolve the fixed Landmateriet grid tiles that positively overlap the ROI."""
    if str(roi.crs) != GRID_CRS:
        raise ValueError(
            f"the national tiling grid is fixed to {GRID_CRS}, but the ROI is in {roi.crs} -- "
            f"reproject the ROI to {GRID_CRS} first (see load_roi)."
        )

    footprint = roi.union_all() if hasattr(roi, "union_all") else roi.unary_union
    tiles = national_grid_tiles(footprint, tile_size)
    if not tiles:
        raise ValueError("ROI tiling produced no tiles; check the ROI geometry and tile_size.")
    return tiles


def tiles_to_geodataframe(tiles: List[Tile], crs: str) -> gpd.GeoDataFrame:
    """Convenience conversion for inspection/debugging (e.g. writing tiles.parquet)."""
    records = {
        "row": [t.row for t in tiles],
        "col": [t.col for t in tiles],
        "tile_id": [t.name for t in tiles],
        "geometry": [t.geometry for t in tiles],
    }
    return gpd.GeoDataFrame(records, crs=crs)


def roi_grid_transform(roi: gpd.GeoDataFrame, grid_size: int) -> Tuple[Affine, Tuple[int, int]]:
    """Compute a grid-aligned raster transform + (height, width) covering the ROI bounds."""
    xmin, ymin, xmax, ymax = roi.total_bounds
    xmin = np.floor(xmin / grid_size) * grid_size
    ymin = np.floor(ymin / grid_size) * grid_size
    xmax = np.ceil(xmax / grid_size) * grid_size
    ymax = np.ceil(ymax / grid_size) * grid_size
    width = int(round((xmax - xmin) / grid_size))
    height = int(round((ymax - ymin) / grid_size))
    transform = from_origin(xmin, ymax, grid_size, grid_size)
    return transform, (height, width)


def rasterize_roi_mask(roi: gpd.GeoDataFrame, transform: Affine, out_shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize the union of all ROI geometries onto a grid; True where covered by the ROI."""
    geom = roi.union_all() if hasattr(roi, "union_all") else roi.unary_union
    mask = rasterize(
        [(geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask.astype(bool)


def region_masks(
    roi: gpd.GeoDataFrame, transform: Affine, out_shape: Tuple[int, int], region_id_field: str
) -> dict:
    """Rasterize each named sub-region of the ROI separately, for per-region statistics."""
    masks = {}
    for name, group in roi.groupby(region_id_field):
        geom = group.union_all() if hasattr(group, "union_all") else group.unary_union
        mask = rasterize(
            [(geom, 1)], out_shape=out_shape, transform=transform, fill=0, dtype="uint8", all_touched=False
        )
        masks[str(name)] = mask.astype(bool)
    return masks
