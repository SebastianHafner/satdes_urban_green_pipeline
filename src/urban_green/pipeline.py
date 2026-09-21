"""End-to-end pipeline: ROI -> DES max-NDVI -> thresholding -> statistics -> change -> report.

This is a single runnable pipeline that wires together ROI handling,
the DES max-NDVI acquisition, and the thresholding methodology into
one multi-year urban green statistics product.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio

from urban_green import acquisition, change, report, statistics, stitching, thresholding
from urban_green.config import PipelineConfig
from urban_green.roi import load_roi, rasterize_roi_mask, region_masks, tile_roi, tiles_to_geodataframe
from urban_green.tile_store import TileStore

logger = logging.getLogger(__name__)


def _build_ndvi_stack(stitched: dict, years, transform, out_shape) -> np.ndarray:
    """Stack each year's stitched mosaic."""
    ndvi_stack = np.full((len(years), *out_shape), np.nan, dtype=np.float32)
    for i, year in enumerate(years):
        with rasterio.open(stitched[year]) as src:
            grid_matches = (src.height, src.width) == out_shape and np.allclose(
                tuple(src.transform), tuple(transform), atol=1e-6
            )
            if not grid_matches:
                raise ValueError(
                    f"Stitched mosaic for year {year} has a different grid than year {years[0]} "
                    f"(({src.height}, {src.width}), {src.transform} vs {out_shape}, {transform}) -- "
                    "DES returned an inconsistent pixel grid across years for the same tiles."
                )
            ndvi_stack[i] = src.read(1).astype(np.float32)
    return ndvi_stack


def run(config: PipelineConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ROI: %s", config.roi_path)
    roi = load_roi(config.roi_path, config.crs)

    logger.info("Resolving national grid tiles (tile_size=%d m)", config.tile_size)
    tiles = tile_roi(roi, config.tile_size)
    tiles_to_geodataframe(tiles, config.crs).to_parquet(config.output_dir / "tiles.parquet")

    logger.info("Connecting to Digital Earth Sweden (%s)", config.des_endpoint)
    connection = acquisition.connect(config)

    logger.info("Tile store: %s", config.tile_store_dir)
    with TileStore(config.tile_store_dir) as tile_store:
        logger.info("Acquiring annual max-NDVI for %d tiles x %d years", len(tiles), len(config.years))
        acquisition.acquire_all(connection, tiles, config.years, config, tile_store)

        stitched_dir = config.output_dir / "stitched"
        logger.info("Stitching tiles per year")
        stitched = stitching.stitch_from_store(tile_store, tiles, config.years, stitched_dir)

    # The working grid follows the tile system directly: it's the actual grid of the
    # stitched mosaic (built from tiles of the fixed national grid), not a separately
    # computed ROI-bbox grid -- so no resampling is needed to line up years or the mask.
    transform, out_shape = stitching.mosaic_grid(stitched[config.years[0]])
    mask = rasterize_roi_mask(roi, transform, out_shape)

    logger.info("Building common NDVI stack across years")
    ndvi_stack = _build_ndvi_stack(stitched, config.years, transform, out_shape)

    logger.info("Thresholding (global GMM + %d-year majority vote)", config.majority_vote_window)
    result = thresholding.run_thresholding(ndvi_stack, mask, window=config.majority_vote_window)

    outputs_dir = config.output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing per-year vegetation classification rasters")
    for i, year in enumerate(config.years):
        layer = result.smoothed_stack[i]
        veg_uint8 = np.where(np.isnan(layer), 255, layer).astype(np.uint8)
        with rasterio.open(
            outputs_dir / f"vegetation_{year}.tif",
            "w",
            driver="GTiff",
            height=veg_uint8.shape[0],
            width=veg_uint8.shape[1],
            count=1,
            dtype="uint8",
            crs=config.crs,
            transform=transform,
            nodata=255,
        ) as dst:
            dst.write(veg_uint8, 1)

    logger.info("Computing change product (loss/gain year)")
    change_years = change.compute_change_years(result.smoothed_stack, config.years)
    change.write_change_geotiff(change_years["loss_year"], transform, config.crs, outputs_dir / "green_loss_year.tif")
    change.write_change_geotiff(change_years["gain_year"], transform, config.crs, outputs_dir / "green_gain_year.tif")

    logger.info("Computing statistics")
    stats = statistics.green_area_stats(result.smoothed_stack, config.years)
    stats.to_csv(outputs_dir / "green_area_statistics.csv", index=False)

    change_stats = statistics.change_summary(change_years["loss_year"], change_years["gain_year"], config.years)
    change_stats.to_csv(outputs_dir / "change_statistics.csv", index=False)

    if config.region_id_field:
        logger.info("Computing per-region statistics (region_id_field=%s)", config.region_id_field)
        masks = region_masks(roi, transform, out_shape, config.region_id_field)
        region_stats = statistics.green_area_stats_by_region(result.smoothed_stack, config.years, masks)
        region_stats.to_csv(outputs_dir / "green_area_statistics_by_region.csv", index=False)

    logger.info("Generating report figures")
    figures_dir = config.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    values = ndvi_stack[:, mask]
    values = values[np.isfinite(values)]
    report.plot_ndvi_distribution(values, result.threshold, figures_dir / "ndvi_distribution.png")
    report.plot_green_area_timeseries(stats, result.threshold, figures_dir / "green_area_timeseries.png")

    logger.info("Pipeline complete. Outputs in %s", outputs_dir)
    return outputs_dir
