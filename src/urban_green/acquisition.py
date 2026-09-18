"""Per-tile, per-year annual max-NDVI acquisition from Digital Earth Sweden.

This replaces satdes_maxndvi/main_server_batch.py's role: same DES query
logic (urban_green.maxndvi.sentinel2/composite/helpers), but driven by an
arbitrary multi-year range and the caller's own ROI tiles instead of a
pre-tiled file and a single hardcoded year. The max-NDVI band is written as
native float16 (no [0, 200] uint8 encoding), and observation count as a
separate uint16 file since a single GeoTIFF cannot mix per-band dtypes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import openeo
import rasterio
from tqdm import tqdm

from urban_green.config import PipelineConfig
from urban_green.maxndvi import composite, helpers, sentinel2

logger = logging.getLogger(__name__)


def connect(config: PipelineConfig):
    connection = openeo.connect(config.des_endpoint)
    connection.authenticate_basic(username=config.des_username, password=config.des_password)
    return connection


def _write_single_band(array: np.ndarray, transform, crs, out_file: Path, dtype: str, nodata) -> None:
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


def acquire_tile_year(connection, tile_geom, tile_id: str, year: int, config: PipelineConfig, out_dir: Path) -> Path:
    """Compute the annual max-NDVI + observation count for one tile/year and write it to disk."""
    ndvi_cube, _ = sentinel2.load_ndvi_cube(connection, tile_geom, config.crs, year, config.cloud_threshold)
    scl_cube = sentinel2.load_scl_cube(connection, tile_geom, config.crs, year, config.cloud_threshold)
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

    ndvi_out = out_dir / f"ndvi_{tile_id}_{year}.tif"
    obs_out = out_dir / f"obs_{tile_id}_{year}.tif"
    _write_single_band(ndvi_arr, transform, crs, ndvi_out, "float16", np.nan)
    _write_single_band(obs_arr, transform, crs, obs_out, "uint16", 0)
    return ndvi_out


def acquire_all(
    connection, tiles: gpd.GeoDataFrame, years, config: PipelineConfig, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        year_dir = out_dir / f"y{year}" / "tiles"
        year_dir.mkdir(parents=True, exist_ok=True)
        for tile in tqdm(list(tiles.itertuples()), desc=f"acquiring {year}"):
            out_file = year_dir / f"ndvi_{tile.tile_id}_{year}.tif"
            if out_file.exists():
                continue  # resume support: skip tiles already acquired
            acquire_tile_year(connection, tile.geometry, tile.tile_id, year, config, year_dir)
