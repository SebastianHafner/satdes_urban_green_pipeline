"""End-to-end test of pipeline.run() with the DES acquisition step mocked out.

Exercises ROI tiling, tile-by-tile processing, the tile-grid-derived working grid,
thresholding, the change product, statistics, and report generation
together - everything except the actual network call to Digital Earth
Sweden.
"""
from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import box

from urban_green import acquisition, thresholding
from urban_green.config import PipelineConfig
from urban_green.pipeline import run

YEARS = [2018, 2019, 2020, 2021, 2022]
GRID_SIZE = 10
SIZE = 10  # 10x10 pixel ROI -> 100m x 100m


def _write_synthetic_tile(out_file: Path, year_index: int) -> None:
    """A 10x10 NDVI tile with a fixed pattern that exercises every code path:
    rows 0-3 always vegetated, rows 4-6 always non-vegetated, row 7 gains
    vegetation at year_index 2, row 8 loses it at year_index 2, row 9 is
    outside all classes (kept non-vegetated) as a control.
    """
    high, low = 0.8, 0.1
    ndvi = np.full((SIZE, SIZE), low, dtype=np.float32)
    ndvi[0:4, :] = high
    ndvi[7, :] = high if year_index >= 2 else low  # gain at index 2
    ndvi[8, :] = low if year_index >= 2 else high  # loss at index 2

    transform = rasterio.transform.from_origin(0, SIZE * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    with rasterio.open(
        out_file, "w", driver="GTiff", height=SIZE, width=SIZE, count=1,
        dtype="float16", crs="EPSG:3006", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(ndvi.astype(np.float16), 1)


def _fake_acquire_all(connection, tiles, years, config, tile_store) -> None:
    tile = tiles[0]
    for i, year in enumerate(years):
        ndvi_path, obs_path = tile_store.paths(tile.row, tile.col, year)
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        _write_synthetic_tile(ndvi_path, i)
        obs_path.write_bytes(b"x")  # obs band is unused downstream; presence is all TileStore checks
        tile_store.register(tile.row, tile.col, year, config.crs, config.cloud_threshold, ndvi_path, obs_path)


def test_pipeline_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "connect", lambda config: None)
    monkeypatch.setattr(acquisition, "acquire_all", _fake_acquire_all)

    roi_path = tmp_path / "roi.gpkg"
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, SIZE * GRID_SIZE, SIZE * GRID_SIZE)]}, crs="EPSG:3006"
    )
    gdf.to_file(roi_path, driver="GPKG")

    config = PipelineConfig(
        roi_path=roi_path,
        years=YEARS,
        output_dir=tmp_path / "out",
        tile_store_dir=tmp_path / "tile_store",
        tile_size=1000,
        majority_vote_window=3,
    )

    outputs_dir = run(config)
    assert outputs_dir == config.output_dir / "outputs"

    # Per-year vegetation classification rasters
    for year in YEARS:
        veg_file = outputs_dir / f"vegetation_{year}.tif"
        assert veg_file.exists()

    # Statistics table: always-vegetated rows should keep ~constant area,
    # and the transition years should show the majority-vote-smoothed effect.
    stats = pd.read_csv(outputs_dir / "green_area_statistics.csv")
    assert list(stats["year"]) == YEARS
    assert (stats["pct_cover"] > 0).all()

    # Change product: gain/loss rasters must exist and encode the expected years.
    with rasterio.open(outputs_dir / "green_gain_year.tif") as src:
        gain = src.read(1)
    with rasterio.open(outputs_dir / "green_loss_year.tif") as src:
        loss = src.read(1)

    # Row 7 (0-indexed from top) gained vegetation at year index 2 -> 2020.
    # Row 8 lost vegetation at year index 2 -> 2020.
    # (rasterio row 0 = top = northernmost = our synthetic row 0)
    assert 2020 in np.unique(gain)
    assert 2020 in np.unique(loss)
    assert (gain[0:4, :] == 0).all()  # always-vegetated rows never "gain"
    assert (loss[4:7, :] == 0).all()  # always-non-vegetated rows never "lose"

    change_stats = pd.read_csv(outputs_dir / "change_statistics.csv")
    row_2020 = change_stats[change_stats["year"] == 2020].iloc[0]
    assert row_2020["loss_km2"] > 0
    assert row_2020["gain_km2"] > 0

    figures_dir = config.output_dir / "figures"
    assert (figures_dir / "ndvi_distribution.png").exists()
    assert (figures_dir / "green_area_timeseries.png").exists()


def test_pipeline_run_with_user_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "connect", lambda config: None)
    monkeypatch.setattr(acquisition, "acquire_all", _fake_acquire_all)

    def fail(*args, **kwargs):
        raise AssertionError("automatic threshold detection should not run")

    monkeypatch.setattr(thresholding, "fit_sampled_threshold", fail)

    roi_path = tmp_path / "roi.gpkg"
    gpd.GeoDataFrame({"geometry": [box(0, 0, SIZE * GRID_SIZE, SIZE * GRID_SIZE)]}, crs="EPSG:3006").to_file(
        roi_path, driver="GPKG"
    )

    # Every synthetic NDVI value (0.1 / 0.8) lies below 0.9, so nothing should be classified as vegetated.
    config = PipelineConfig(
        roi_path=roi_path,
        years=YEARS,
        output_dir=tmp_path / "out",
        tile_store_dir=tmp_path / "tile_store",
        tile_size=1000,
        ndvi_threshold=0.9,
    )

    outputs_dir = run(config)
    stats = pd.read_csv(outputs_dir / "green_area_statistics.csv")
    assert (stats["pct_cover"] == 0).all()


def test_config_rejects_out_of_bounds_ndvi_threshold(tmp_path):
    with pytest.raises(ValueError, match="ndvi_threshold"):
        PipelineConfig(roi_path=tmp_path / "roi.gpkg", years=YEARS, output_dir=tmp_path, ndvi_threshold=1.5)


def test_config_rejects_too_small_gmm_sample_size(tmp_path):
    with pytest.raises(ValueError, match="gmm_sample_size"):
        PipelineConfig(roi_path=tmp_path / "roi.gpkg", years=YEARS, output_dir=tmp_path, gmm_sample_size=1)


TILE_PX = 10  # tile_size=100 m at 10 m pixels


def _fake_acquire_random_tiles(connection, tiles, years, config, tile_store) -> None:
    """Random NDVI per tile and year, each raster exactly covering its tile's bounds."""
    for tile in tiles:
        rng = np.random.default_rng(tile.row * 100 + tile.col)
        for year in years:
            ndvi_path, obs_path = tile_store.paths(tile.row, tile.col, year)
            obs_path.parent.mkdir(parents=True, exist_ok=True)
            minx, _, _, maxy = tile.bounds
            ndvi = rng.uniform(0.0, 0.9, size=(TILE_PX, TILE_PX)).astype(np.float16)
            ndvi[0, 0] = np.nan  # some nodata
            with rasterio.open(
                ndvi_path, "w", driver="GTiff", height=TILE_PX, width=TILE_PX, count=1, dtype="float16",
                crs="EPSG:3006", transform=rasterio.transform.from_origin(minx, maxy, GRID_SIZE, GRID_SIZE),
                nodata=np.nan,
            ) as dst:
                dst.write(ndvi, 1)
            obs_path.write_bytes(b"x")
            tile_store.register(tile.row, tile.col, year, config.crs, config.cloud_threshold, ndvi_path, obs_path)


def test_tile_by_tile_run_matches_whole_array_computation(tmp_path, monkeypatch):
    """Processing tile by tile must give exactly the rasters and statistics of the
    whole-array functions applied to the full mosaic, including an empty grid cell."""
    from urban_green import change, statistics
    from urban_green.roi import load_roi, rasterize_roi_mask, region_masks

    monkeypatch.setattr(acquisition, "connect", lambda config: None)
    monkeypatch.setattr(acquisition, "acquire_all", _fake_acquire_random_tiles)

    # L-shaped ROI over tiles (row, col) = (0, 0), (0, 1), (1, 0); tile (1, 1) stays empty.
    # Region "b" spans two tiles and only partly covers them.
    roi_path = tmp_path / "roi.gpkg"
    gpd.GeoDataFrame(
        {"name": ["a", "b"], "geometry": [box(5, 5, 95, 195), box(95, 15, 185, 85)]}, crs="EPSG:3006"
    ).to_file(roi_path, driver="GPKG")

    threshold = 0.45
    config = PipelineConfig(
        roi_path=roi_path, years=YEARS, output_dir=tmp_path / "out", tile_store_dir=tmp_path / "tile_store",
        tile_size=100, ndvi_threshold=threshold, region_id_field="name",
    )
    outputs_dir = run(config)

    # Whole-array reference on the full 20x20 mosaic (VRT), as the pipeline used to compute it.
    stack = []
    for year in YEARS:
        with rasterio.open(config.output_dir / "stitched" / f"maxndvi_{year}.vrt") as src:
            stack.append(src.read(1).astype(np.float32))
            transform, shape = src.transform, src.shape
    stack = np.stack(stack)
    assert shape == (20, 20)
    assert np.isnan(stack[:, :TILE_PX, TILE_PX:]).all()  # empty tile (1, 1) is the top-right cell

    roi = load_roi(roi_path, config.crs)
    mask = rasterize_roi_mask(roi, transform, shape)
    expected = thresholding.run_thresholding(stack, mask, window=3, threshold=threshold).smoothed_stack
    expected_change = change.compute_change_years(expected, YEARS)

    for i, year in enumerate(YEARS):
        with rasterio.open(outputs_dir / f"vegetation_{year}.tif") as src:
            np.testing.assert_array_equal(src.read(1), np.where(np.isnan(expected[i]), 255, expected[i]).astype(np.uint8))
    for name, key in (("green_loss_year.tif", "loss_year"), ("green_gain_year.tif", "gain_year")):
        with rasterio.open(outputs_dir / name) as src:
            np.testing.assert_array_equal(src.read(1), expected_change[key])

    pd.testing.assert_frame_equal(
        pd.read_csv(outputs_dir / "green_area_statistics.csv"),
        pd.read_csv(io.StringIO(statistics.green_area_stats(expected, YEARS).to_csv(index=False))),
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(outputs_dir / "change_statistics.csv"),
        pd.read_csv(io.StringIO(
            statistics.change_summary(expected_change["loss_year"], expected_change["gain_year"], YEARS).to_csv(index=False)
        )),
    )
    masks = region_masks(roi, transform, shape, "name")
    pd.testing.assert_frame_equal(
        pd.read_csv(outputs_dir / "green_area_statistics_by_region.csv"),
        pd.read_csv(io.StringIO(
            statistics.green_area_stats_by_region(expected, YEARS, masks).to_csv(index=False)
        )),
    )
