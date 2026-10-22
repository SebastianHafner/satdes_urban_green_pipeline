"""End-to-end pipeline: ROI -> DES max-NDVI -> thresholding -> statistics -> change -> report.

This is a single runnable pipeline that wires together ROI handling,
the DES max-NDVI acquisition, and the thresholding methodology into
one multi-year urban green statistics product.
"""
from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio import windows
from tqdm import tqdm

from urban_green import acquisition, change, report, statistics, stitching, thresholding
from urban_green.config import PipelineConfig
from urban_green.roi import (
    load_roi,
    rasterize_geometry,
    region_geometries,
    roi_geometry,
    tile_roi,
    tiles_to_geodataframe,
)
from urban_green.tile_store import TileStore

logger = logging.getLogger(__name__)


def _boxes_intersect(a, b) -> bool:
    """Whether two (minx, miny, maxx, maxy) boxes overlap with positive area."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _iter_tiles(tile_paths: stitching.TilePaths, years, transform, roi_geom, desc: str):
    """Yield (ndvi_stack, roi_mask, window) per tile, one tile (all years) in memory at a time."""
    for year_paths in tqdm(tile_paths.values(), desc=desc, unit="tile"):
        ndvi_stack, window = stitching.read_tile_stack(year_paths, years, transform)
        mask = rasterize_geometry(roi_geom, windows.transform(window, transform), ndvi_stack.shape[1:])
        yield ndvi_stack, mask, window


def _create_raster(path: Path, out_shape, transform, crs: str, dtype: str, nodata):
    # Pixels outside every tile are never written; GDAL fills them with `nodata`.
    return rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out_shape[0],
        width=out_shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    )


def run(config: PipelineConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ROI: %s", config.roi_path)
    roi = load_roi(config.roi_path, config.crs)

    # Preparing grid
    logger.info("Resolving national grid tiles (tile_size=%d m)", config.tile_size)
    tiles = tile_roi(roi, config.tile_size)
    tiles_to_geodataframe(tiles, config.crs).to_parquet(config.output_dir / "tiles.parquet")

    # Downloading maximum NDVI product from Digital Earth Sweden
    logger.info("Connecting to Digital Earth Sweden (%s)", config.des_endpoint)
    connection = acquisition.connect(config)

    logger.info("Tile store: %s", config.tile_store_dir)
    with TileStore(config.tile_store_dir) as tile_store:
        logger.info("Acquiring annual max-NDVI for %d tiles x %d years", len(tiles), len(config.years))
        acquisition.acquire_all(connection, tiles, config.years, config, tile_store)
        tile_paths = stitching.resolve_tile_paths(tile_store, tiles, config.years)

    # Create ROI-wide products from tiles
    transform, out_shape = stitching.tile_grid([paths[config.years[0]] for paths in tile_paths.values()])

    stitched_dir = config.output_dir / "stitched"
    logger.info("Writing per-year VRT mosaics to %s", stitched_dir)
    stitching.write_year_vrts(tile_paths, config.years, transform, out_shape, config.crs, stitched_dir)

    roi_geom = roi_geometry(roi)

    # Pass 1: sample NDVI inside the ROI, for the GMM threshold and the distribution figure.
    logger.info("Sampling NDVI inside the ROI (%d tiles)", len(tile_paths))
    sampler = thresholding.ValueSampler(config.gmm_sample_size)
    for ndvi_stack, mask, _ in _iter_tiles(tile_paths, config.years, transform, roi_geom, "sampling"):
        sampler.add(ndvi_stack[:, mask])

    if config.ndvi_threshold is None:
        logger.info("Thresholding (global GMM + %d-year majority vote)", config.majority_vote_window)
        threshold = thresholding.fit_sampled_threshold(sampler)
        thresholding.check_threshold_range(threshold)
    else:
        logger.info(
            "Thresholding (user-defined threshold %.3f + %d-year majority vote)",
            config.ndvi_threshold,
            config.majority_vote_window,
        )
        threshold = config.ndvi_threshold

    outputs_dir = config.output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    regions = region_geometries(roi, config.region_id_field) if config.region_id_field else {}
    n_years = len(config.years)
    n_valid = np.zeros(n_years, dtype=np.int64)
    n_green = np.zeros(n_years, dtype=np.int64)
    loss_counts = np.zeros(n_years - 1, dtype=np.int64)
    gain_counts = np.zeros(n_years - 1, dtype=np.int64)
    region_counts = {name: (np.zeros(n_years, dtype=np.int64), np.zeros(n_years, dtype=np.int64)) for name in regions}

    # Pass 2: classify, smooth and derive the change product per tile; write each tile's
    # window of the output rasters and accumulate the statistics as pixel counts.
    logger.info("Classifying, writing rasters and computing statistics per tile")
    with ExitStack() as stack:
        veg_dsts = [
            stack.enter_context(
                _create_raster(outputs_dir / f"vegetation_{year}.tif", out_shape, transform, config.crs, "uint8", 255)
            )
            for year in config.years
        ]
        loss_dst, gain_dst = (
            stack.enter_context(_create_raster(outputs_dir / name, out_shape, transform, config.crs, "uint16", 0))
            for name in ("green_loss_year.tif", "green_gain_year.tif")
        )

        for ndvi_stack, mask, window in _iter_tiles(tile_paths, config.years, transform, roi_geom, "classifying"):
            result = thresholding.run_thresholding(
                ndvi_stack, mask, window=config.majority_vote_window, threshold=threshold
            )
            smoothed = result.smoothed_stack
            for i, dst in enumerate(veg_dsts):
                dst.write(np.where(np.isnan(smoothed[i]), 255, smoothed[i]).astype(np.uint8), 1, window=window)

            change_years = change.compute_change_years(smoothed, config.years)
            loss_dst.write(change_years["loss_year"], 1, window=window)
            gain_dst.write(change_years["gain_year"], 1, window=window)

            tile_valid, tile_green = statistics.green_counts(smoothed)
            n_valid += tile_valid
            n_green += tile_green
            loss_counts += statistics.change_counts(change_years["loss_year"], config.years)
            gain_counts += statistics.change_counts(change_years["gain_year"], config.years)

            window_transform = windows.transform(window, transform)
            tile_bounds = windows.bounds(window, transform)
            for name, geom in regions.items():
                if not _boxes_intersect(geom.bounds, tile_bounds):
                    continue
                region_mask = rasterize_geometry(geom, window_transform, mask.shape)
                region_valid, region_green = statistics.green_counts(
                    np.where(region_mask[None, :, :], smoothed, np.nan)
                )
                region_counts[name][0][:] += region_valid
                region_counts[name][1][:] += region_green

    stats = statistics.green_area_stats_from_counts(n_valid, n_green, config.years)
    stats.to_csv(outputs_dir / "green_area_statistics.csv", index=False)

    change_stats = statistics.change_summary_from_counts(loss_counts, gain_counts, config.years)
    change_stats.to_csv(outputs_dir / "change_statistics.csv", index=False)

    if config.region_id_field:
        logger.info("Writing per-region statistics (region_id_field=%s)", config.region_id_field)
        region_stats = statistics.green_area_stats_by_region_from_counts(region_counts, config.years)
        region_stats.to_csv(outputs_dir / "green_area_statistics_by_region.csv", index=False)

    logger.info("Generating report figures")
    figures_dir = config.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report.plot_ndvi_distribution(sampler.sample, threshold, figures_dir / "ndvi_distribution.png")
    report.plot_green_area_timeseries(stats, threshold, figures_dir / "green_area_timeseries.png")

    logger.info("Pipeline complete. Outputs in %s", outputs_dir)
    return outputs_dir
