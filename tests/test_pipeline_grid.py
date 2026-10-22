"""Unit tests for the tile-grid-derived working grid: stitching.read_tile_stack must
stack years directly (no resampling) when a tile's rasters share a grid, and must
raise loudly rather than silently resample when they don't.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio

from urban_green.stitching import read_tile_stack, tile_grid, write_year_vrt

SHAPE = (10, 10)


def _write_raster(path, transform, shape=SHAPE, value=0.0):
    with rasterio.open(
        path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
        dtype="float32", crs="EPSG:3006", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(np.full(shape, value, dtype=np.float32), 1)


def test_tile_grid_covers_union_of_tiles(tmp_path):
    # Two 10x10 tiles diagonally apart -> 20x20 grid with two empty quadrants.
    _write_raster(tmp_path / "a.tif", rasterio.transform.from_origin(0, 200, 10, 10))
    _write_raster(tmp_path / "b.tif", rasterio.transform.from_origin(100, 100, 10, 10))
    transform, shape = tile_grid([tmp_path / "a.tif", tmp_path / "b.tif"])
    assert shape == (20, 20)
    assert tuple(transform) == tuple(rasterio.transform.from_origin(0, 200, 10, 10))


def test_read_tile_stack_reads_directly_when_grids_match(tmp_path):
    transform = rasterio.transform.from_origin(100, 100, 10, 10)
    paths = {}
    for i, year in enumerate([2020, 2021]):
        paths[year] = tmp_path / f"y{year}.tif"
        _write_raster(paths[year], transform, value=float(i))

    grid = rasterio.transform.from_origin(0, 200, 10, 10)
    stack, window = read_tile_stack(paths, [2020, 2021], grid)
    assert stack.shape == (2, *SHAPE)
    assert np.allclose(stack[0], 0.0)
    assert np.allclose(stack[1], 1.0)
    assert (window.col_off, window.row_off, window.width, window.height) == (10, 10, 10, 10)


def test_read_tile_stack_raises_on_shifted_origin(tmp_path):
    transform = rasterio.transform.from_origin(0, 100, 10, 10)
    shifted = rasterio.transform.from_origin(5, 100, 10, 10)  # half-pixel shift

    paths = {2020: tmp_path / "y2020.tif", 2021: tmp_path / "y2021.tif"}
    _write_raster(paths[2020], transform)
    _write_raster(paths[2021], shifted)

    with pytest.raises(ValueError, match="inconsistent pixel grid"):
        read_tile_stack(paths, [2020, 2021], transform)


def test_read_tile_stack_raises_on_shape_mismatch(tmp_path):
    transform = rasterio.transform.from_origin(0, 100, 10, 10)
    paths = {2020: tmp_path / "y2020.tif", 2021: tmp_path / "y2021.tif"}
    _write_raster(paths[2020], transform, shape=SHAPE)
    _write_raster(paths[2021], transform, shape=(5, 5))

    with pytest.raises(ValueError, match="inconsistent pixel grid"):
        read_tile_stack(paths, [2020, 2021], transform)


def test_read_tile_stack_raises_on_tile_off_working_grid(tmp_path):
    path = tmp_path / "y2020.tif"
    _write_raster(path, rasterio.transform.from_origin(5, 100, 10, 10))
    with pytest.raises(ValueError, match="not aligned"):
        read_tile_stack({2020: path}, [2020], rasterio.transform.from_origin(0, 100, 10, 10))


def test_write_year_vrt_mosaics_tiles(tmp_path):
    _write_raster(tmp_path / "a.tif", rasterio.transform.from_origin(0, 200, 10, 10), value=1.0)
    _write_raster(tmp_path / "b.tif", rasterio.transform.from_origin(100, 100, 10, 10), value=2.0)
    files = [tmp_path / "a.tif", tmp_path / "b.tif"]
    transform, shape = tile_grid(files)
    vrt = write_year_vrt(files, transform, shape, "EPSG:3006", tmp_path / "mosaic.vrt")

    with rasterio.open(vrt) as src:
        assert src.shape == shape
        assert tuple(src.transform) == tuple(transform)
        assert src.crs.to_epsg() == 3006
        data = src.read(1)
    assert np.all(data[:10, :10] == 1.0)
    assert np.all(data[10:, 10:] == 2.0)
    assert np.all(np.isnan(data[:10, 10:]))  # no tile there
