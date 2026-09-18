"""Region-of-interest ingestion: load, tile, and rasterize an arbitrary Swedish ROI.

Tiling logic is adapted from satdes_maxndvi/tiling.py (generalized to a
function, and restricted to tiles that actually intersect the ROI geometry
instead of tiling the full bounding-box grid).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from shapely.geometry import box


def load_roi(roi_path: Path, crs: str) -> gpd.GeoDataFrame:
    """Load a user-supplied ROI vector file and reproject it to the working CRS."""
    roi_path = Path(roi_path)
    gdf = gpd.read_parquet(roi_path) if roi_path.suffix == ".parquet" else gpd.read_file(roi_path)
    if gdf.crs is None:
        raise ValueError(f"ROI file {roi_path} has no CRS defined; cannot reproject safely.")
    if len(gdf) == 0:
        raise ValueError(f"ROI file {roi_path} contains no features.")
    return gdf.to_crs(crs)


def tile_roi(roi: gpd.GeoDataFrame, tile_size: int, region_id_field: Optional[str] = None) -> gpd.GeoDataFrame:
    """Split each ROI feature into a grid of tile_size x tile_size boxes.

    Only tiles that intersect the feature's geometry are kept (unlike the
    original tiling.py, which tiled the full bounding box regardless of the
    polygon's shape).
    """
    records = {"name": [], "tile_j": [], "tile_i": [], "tile_id": [], "geometry": []}
    for idx, row in roi.iterrows():
        name = str(row[region_id_field]) if region_id_field else str(idx)
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        # ceil (not int(...)+1, satdes_maxndvi/tiling.py's original formula) avoids
        # an extra degenerate zero-width/height tile when a dimension is exactly
        # divisible by tile_size.
        n = max(1, math.ceil((maxx - minx) / tile_size))
        m = max(1, math.ceil((maxy - miny) / tile_size))

        for j in range(n):
            for i in range(m):
                minx_tile = minx + j * tile_size
                maxy_tile = maxy - i * tile_size
                maxx_tile = maxx if j == n - 1 else minx_tile + tile_size
                miny_tile = miny if i == m - 1 else maxy_tile - tile_size
                tile_geom = box(minx_tile, miny_tile, maxx_tile, maxy_tile)

                # area > 0 (not just .intersects()) excludes tiles that only touch the
                # ROI boundary with zero overlapping area, which would otherwise trigger
                # a wasted DES acquisition call for a tile with nothing to cover.
                if tile_geom.intersection(geom).area <= 0:
                    continue

                records["name"].append(name)
                records["tile_j"].append(j)
                records["tile_i"].append(i)
                records["tile_id"].append(f"{name}_{j}_{i}")
                records["geometry"].append(tile_geom)

    if not records["geometry"]:
        raise ValueError("ROI tiling produced no tiles; check the ROI geometry and tile_size.")

    return gpd.GeoDataFrame(records, crs=roi.crs)


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
