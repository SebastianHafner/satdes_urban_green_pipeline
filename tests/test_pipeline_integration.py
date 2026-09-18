"""End-to-end test of pipeline.run() with the DES acquisition step mocked out.

Exercises ROI tiling, stitching, the common-grid reprojection, thresholding,
the change product, statistics, and report generation together - everything
except the actual network call to Digital Earth Sweden.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from urban_green import acquisition
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


def _fake_acquire_all(connection, tiles, years, config, out_dir) -> None:
    tile_id = list(tiles.itertuples())[0].tile_id
    for i, year in enumerate(years):
        year_dir = out_dir / f"y{year}" / "tiles"
        year_dir.mkdir(parents=True, exist_ok=True)
        _write_synthetic_tile(year_dir / f"ndvi_{tile_id}_{year}.tif", i)


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
        tile_size=1000,
        grid_size=GRID_SIZE,
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
