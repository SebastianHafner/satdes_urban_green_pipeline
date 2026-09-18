# satdes_urban_green_pipeline

A runnable, multi-year urban green statistics pipeline for Sweden. This is
the demonstrator for **SATDES Task 1, Subtask 1.6** ("Setting up, testing
and evaluating an interaction model ... to query and aggregate data to
statistical result in a single pipeline"). It is not delivered as an API;
instead it's a CLI and a thin Jupyter notebook front end over the same
Python package, so it can be run end-to-end on an arbitrary region of
interest.

It wires together:

- **Acquisition** — annual Sentinel-2 max-NDVI + observation-count
  composites from [Digital Earth Sweden](https://digitalearth.se), using
  the same openEO query logic as
  [`satdes_maxndvi`](../satdes_maxndvi) (`main_server_batch.py`),
  generalized from a single hardcoded year to an arbitrary year range and
  from a pre-tiled file to the caller's own ROI.
- **Thresholding** — the vegetated/non-vegetated classification method
  recommended in the Subtask 1.5 (Overleaf) thresholding report: a single
  Gaussian Mixture Model fit on urban-masked NDVI pixels pooled across all
  years, thresholded at the analytic intersection of the two fitted
  Gaussians, followed by a 3-year symmetric majority-vote smoothing of the
  per-year binary classification.
- **Statistics & change detection** — annual green area (km²) and % cover,
  year-on-year change, and a change-year product (see below).

## Installation

```bash
pip install -e .
```

Requires GDAL ≥ 3.11 (via `rasterio`) for native `float16` GeoTIFF support,
used for the max-NDVI rasters.

## Usage

### CLI

```bash
python -m urban_green.cli \
    --roi path/to/region_of_interest.gpkg \
    --years 2018:2025 \
    --out ./outputs
```

Any GDAL/OGR-readable vector format is accepted for `--roi` (Shapefile,
GeoPackage, GeoParquet, GeoJSON, ...). `--years` accepts either an
inclusive range (`2018:2025`) or an explicit comma list (`2018,2020,2022`).
Pass `--region-id-field <column>` to also get per-named-sub-region
statistics when the ROI has multiple features (e.g. several municipalities
or tätorter).

### Notebook

`notebooks/urban_green_demo.ipynb` is a thin GUI front end over the same
package: preview the ROI, run the pipeline, and inspect the resulting
figures and tables inline.

### Credentials

Digital Earth Sweden access uses openEO Basic Auth. Set:

```bash
export DES_USERNAME=your_username
export DES_PASSWORD=your_password
```

If unset, the pipeline falls back to `satdes_maxndvi`'s placeholder test
credentials (`testuser` / `secretpassword`) so the demonstrator still runs
out of the box against whatever access those credentials provide.

## Outputs

Written under `<output_dir>/outputs/`:

| File | Description |
|---|---|
| `vegetation_<year>.tif` | Per-year smoothed binary vegetation classification (uint8, 0/1, nodata=255) |
| `green_loss_year.tif` | Year of the **first** vegetated→non-vegetated transition per pixel (uint16, fill=0 = no loss) |
| `green_gain_year.tif` | Year of the **first** non-vegetated→vegetated transition per pixel (uint16, fill=0 = no gain) |
| `green_area_statistics.csv` | Annual green area (km²), % cover, and year-on-year change |
| `change_statistics.csv` | Annual green loss/gain area (km²) |
| `green_area_statistics_by_region.csv` | Same as above, broken out per named sub-region (only if `--region-id-field` was given) |

Figures matching the Overleaf thresholding report's style are written under
`<output_dir>/figures/` (NDVI distribution with threshold overlay, green
area time series with year-on-year change bars).

Intermediate per-tile and per-year mosaic rasters are kept under
`<output_dir>/acquisition/` and `<output_dir>/stitched/` for inspection or
reuse across runs (acquisition is resumable: existing tile files are
skipped).

## Design notes

- The max-NDVI band is stored as native `float16` (no `[0, 200]` uint8
  encoding/decoding round-trip, which was a source of a real inconsistency
  found across the exploratory notebooks: two different decode formulas
  were in use). Observation count is written as a separate `uint16` file
  since a single GeoTIFF cannot mix per-band dtypes.
- `satdes_maxndvi`'s DES-facing query logic
  (`data_loading/sentinel2.py`, `data_loading/helpers.py`,
  `methods/composite.py`) is vendored (trimmed, unmodified query logic)
  under `src/urban_green/maxndvi/`.
- ROI tiling only keeps tiles with positive-area intersection with the ROI
  (not just `.intersects()`, which also matches tiles that merely touch the
  ROI boundary with zero overlapping area) — avoids wasted DES acquisition
  calls for tiles with nothing to cover.
- The change-year product encodes the **first** transition per pixel per
  direction (matching the convention of products like Hansen Global Forest
  Change's "loss year"); loss and gain are reported as separate rasters
  rather than mixed into one signed/banded product.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite covers the thresholding, change-detection, statistics, and
ROI-tiling logic directly, plus one end-to-end integration test that runs
the full pipeline with the DES acquisition step mocked out (everything
except the live network call).
