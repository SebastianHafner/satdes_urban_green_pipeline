"""Per-tile, per-year annual max-NDVI acquisition from Digital Earth Sweden.

The max-NDVI band is written as native float16, and observation count as a
separate uint16 file.

Tiles are resolved against the Landmateriet grid and cached in a shared
tile store, so a tile already acquired by a previous run  is reused instead
of being re-downloaded.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import openeo
import rasterio
from tqdm import tqdm

from urban_green.config import PipelineConfig
from urban_green.grid import Tile
from urban_green.maxndvi import composite, helpers, sentinel2
from urban_green.tile_store import TileStore

logger = logging.getLogger(__name__)


def connect(config: PipelineConfig):
    connection = openeo.connect(config.des_endpoint)
    connection.authenticate_basic(username=config.des_username, password=config.des_password)
    return connection


def _write_single_band(array: np.ndarray, transform, crs, out_file: Path, dtype: str, nodata) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def acquire_tile_year(connection, tile: Tile, year: int, config: PipelineConfig, ndvi_out: Path, obs_out: Path) -> None:
    """Compute the annual max-NDVI + observation count for one tile/year and write it to disk."""
    ndvi_cube, _ = sentinel2.load_ndvi_cube(connection, tile.geometry, config.crs, year, config.cloud_threshold)
    scl_cube = sentinel2.load_scl_cube(connection, tile.geometry, config.crs, year, config.cloud_threshold)
    ndvi_masked_cube = sentinel2.cloud_masking(ndvi_cube, scl_cube)

    valid_obs_count = ndvi_masked_cube.reduce_dimension(dimension="t", reducer="count")
    max_ndvi = composite.max_composite_server(ndvi_masked_cube)

    # Band order fixed by this merge: index 0 = max_ndvi, index 1 = valid_obs_count.
    max_ndvi = max_ndvi.add_dimension(name="bands", label="max_ndvi", type="bands")
    valid_obs_count = valid_obs_count.add_dimension(name="bands", label="valid_obs_count", type="bands")
    res_datacube = max_ndvi.merge_cubes(valid_obs_count)

    result = helpers.run_job(res_datacube, single_image=True)
    image = helpers.extract_image(result)

    transform = image.rio.transform()
    crs = image.rio.crs or config.crs

    values = image.values  # (2, H, W)
    ndvi_arr = values[0].astype(np.float16)
    obs_arr = np.nan_to_num(values[1], nan=0).astype(np.uint16)

    _write_single_band(ndvi_arr, transform, crs, ndvi_out, "float16", np.nan)
    _write_single_band(obs_arr, transform, crs, obs_out, "uint16", 0)


def acquire_all(connection, tiles: List[Tile], years, config: PipelineConfig, tile_store: TileStore) -> None:
    for year in years:
        for tile in tqdm(tiles, desc=f"acquiring {year}"):
            if not config.force_reacquire and tile_store.is_available(tile.row, tile.col, year):
                continue  # cached: reuse the tile already acquired by this or a previous run

            ndvi_out, obs_out = tile_store.paths(tile.row, tile.col, year)
            acquire_tile_year(connection, tile, year, config, ndvi_out, obs_out)
            tile_store.register(tile.row, tile.col, year, config.crs, config.cloud_threshold, ndvi_out, obs_out)
