"""Unit tests for the tile-grid-derived working grid: pipeline._build_ndvi_stack
must stack years directly (no resampling) when their mosaics share a grid,
and must raise loudly rather than silently resample when they don't.
"""
from __future__ import annotations

import numpy as np
import rasterio

from urban_green.pipeline import _build_ndvi_stack
from urban_green.stitching import mosaic_grid

SHAPE = (10, 10)


def _write_raster(path, transform, shape=SHAPE, value=0.0):
    with rasterio.open(
        path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
        dtype="float32", crs="EPSG:3006", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(np.full(shape, value, dtype=np.float32), 1)


def test_mosaic_grid_reads_transform_and_shape(tmp_path):
    transform = rasterio.transform.from_origin(0, 100, 10, 10)
    path = tmp_path / "y2020.tif"
    _write_raster(path, transform)
    read_transform, shape = mosaic_grid(path)
    assert shape == SHAPE
    assert tuple(read_transform) == tuple(transform)


def test_build_ndvi_stack_reads_directly_when_grids_match(tmp_path):
    transform = rasterio.transform.from_origin(0, 100, 10, 10)
    paths = {}
    for i, year in enumerate([2020, 2021]):
        path = tmp_path / f"y{year}.tif"
        _write_raster(path, transform, value=float(i))
        paths[year] = path

    stack = _build_ndvi_stack(paths, [2020, 2021], transform, SHAPE)
    assert stack.shape == (2, *SHAPE)
    assert np.allclose(stack[0], 0.0)
    assert np.allclose(stack[1], 1.0)


def test_build_ndvi_stack_raises_on_shifted_origin(tmp_path):
    transform = rasterio.transform.from_origin(0, 100, 10, 10)
    shifted = rasterio.transform.from_origin(5, 100, 10, 10)  # half-pixel shift

    paths = {2020: tmp_path / "y2020.tif", 2021: tmp_path / "y2021.tif"}
    _write_raster(paths[2020], transform)
    _write_raster(paths[2021], shifted)

    try:
        _build_ndvi_stack(paths, [2020, 2021], transform, SHAPE)
        assert False, "expected ValueError for a shifted-origin grid mismatch"
    except ValueError as exc:
        assert "inconsistent pixel grid" in str(exc)


def test_build_ndvi_stack_raises_on_shape_mismatch(tmp_path):
    transform = rasterio.transform.from_origin(0, 100, 10, 10)
    paths = {2020: tmp_path / "y2020.tif", 2021: tmp_path / "y2021.tif"}
    _write_raster(paths[2020], transform, shape=SHAPE)
    _write_raster(paths[2021], transform, shape=(5, 5))

    try:
        _build_ndvi_stack(paths, [2020, 2021], transform, SHAPE)
        assert False, "expected ValueError for a shape mismatch"
    except ValueError:
        pass
