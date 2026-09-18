"""Vendored from satdes_maxndvi/data_loading/sentinel2.py (unchanged logic).

These are the pure, side-effect-free openEO process-graph builders that
satdes_maxndvi/main_server_batch.py drives directly. Kept as-is so the
DES-facing query logic stays consistent with the upstream repo.
"""
from __future__ import annotations

from typing import Tuple

from shapely.geometry.base import BaseGeometry

from urban_green.maxndvi.helpers import get_bbox, get_end_date, get_start_date


def load_ndvi_cube(conn, geom: BaseGeometry, crs: str, year: int, cloud_cover_threshold: int) -> Tuple:
    s2_cube = load_s2_cube(conn, geom, crs, year, cloud_cover_threshold)
    ndvi_cube = s2_cube.ndvi(nir="b08", red="b04")
    return ndvi_cube, s2_cube


def load_s2_cube(
    conn,
    geom: BaseGeometry,
    crs: str,
    year: int,
    cloud_cover_threshold: int,
    bands: Tuple[str, ...] = ("b04", "b08"),
):
    bbox = get_bbox(geom, crs)
    s2_cube = conn.load_collection(
        "s2_msi_l2a",
        spatial_extent=bbox,
        temporal_extent=[get_start_date(year), get_end_date(year)],
        bands=list(bands),
        properties={"eo:cloud_cover": lambda val: val < cloud_cover_threshold},
    )
    # Harmonization and scaling: https://clearsky.vision/knowledge/sentinel2-scaling-harmonization
    s2_cube = s2_cube.subtract(1_000)  # undo the -1000 DN offset
    s2_cube = s2_cube.apply(lambda band: band.clip(min=0, max=10_000))
    s2_cube = s2_cube.divide(10_000)  # scale to 0.0-1.0 reflectance
    return s2_cube


def load_scl_cube(conn, geom: BaseGeometry, crs: str, year: int, cloud_cover_threshold: int):
    bbox = get_bbox(geom, crs)
    return conn.load_collection(
        "s2_msi_l2a",
        spatial_extent=bbox,
        temporal_extent=[get_start_date(year), get_end_date(year)],
        bands=["scl"],
        properties={"eo:cloud_cover": lambda val: val < cloud_cover_threshold},
    )


def cloud_masking(data_cube, scl_cube, mask_perpetual_snow: bool = False):
    scl_band = scl_cube.band("scl").resample_cube_spatial(data_cube)

    cloud_mask = (scl_band == 8) | (scl_band == 9) | (scl_band == 10)
    snow_mask = scl_band == 11

    if not mask_perpetual_snow:
        snow_mask_cloud_free = snow_mask.mask(cloud_mask)
        perpetual_snow = snow_mask_cloud_free.reduce_dimension(dimension="t", reducer="all")
        not_perpetual_snow = perpetual_snow == 0
        snow_mask = snow_mask.merge_cubes(not_perpetual_snow, overlap_resolver="and")

    mask = (cloud_mask == 1) | (snow_mask == 1)
    return data_cube.mask(mask)
