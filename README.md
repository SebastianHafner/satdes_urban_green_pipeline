# SATDES - Experimental Pipeline for Urban Green Statistic Production

This demonstrator consists of a command line interface (CLI) and a Jupyter notebook front end using the same Python package.It can produce multi-year urban greeen statistics for an arbitrary region of interest within Sweden.

It wires together:

- **Acquisition** — annual Sentinel-2 max-NDVI composites from [Digital Earth Sweden](https://digitalearth.se).
- **Tiling** — a fixed grid in SWEREF99 TM (EPSG:3006), matching Lantmäteriet's national index grid convention (10 km cells).
- **Tile store** — a shared, cross-run cache of acquired tiles. Before acquiring a tile/year, the pipeline checks whether it's already cached and skips the acquisition call if so.
- **Thresholding** — a vegetated/non-vegetated classification method based on a Gaussian Mixture Model fit on NDVI pixels across all years, thresholded at the intersection of the two fitted Gaussians, followed by a 3-year symmetric majority-vote smoothing of the per-year binary classification.
- **Statistics & change detection** — annual green area (km²) and % cover, year-on-year change, and a change-year product.

## Installation

Requires Python ≥ 3.10 and GDAL ≥ 3.11 (via `rasterio`).

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e .               # or `pip install -e ".[dev]"` to also get pytest
```

Verify the install:

```bash
python -m urban_green.cli --help
```

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

`tests/data/tatorter_2023_orebro.parquet` (the Örebro tätort boundary,
EPSG:3006) is bundled as a ready-made example ROI — the same one the
Overleaf Subtask 1.5 report validates against — useful for a quick first
run or for manual/debug testing:

```bash
python -m urban_green.cli \
    --roi tests/data/tatorter_2023_orebro.parquet \
    --years 2022:2025 \
    --out ./outputs
```

### Tile store (acquisition cache)

`--tile-store <path>` (default `./tile_store`) points at a shared cache of
acquired NDVI tiles, reused across runs and ROIs — run the pipeline again
over an overlapping or adjacent ROI and any tiles already in the store are
reused instead of re-downloaded. To bypass the cache, use `--force-reacquire`
, or point `--tile-store` at a fresh directory.

### Notebook

`notebooks/urban_green_demo.ipynb` is a thin GUI front end over the same
package: preview the ROI, run the pipeline, and inspect the resulting
figures and tables inline.

### Credentials

The pipeline uses our test credentials (`testuser` / `secretpassword`) to access Digital Earth Sweden (openEO Basic Auth.).

To use your own credentials, set:

```bash
export DES_USERNAME=your_username
export DES_PASSWORD=your_password
```

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



Per-tile NDVI rasters live in the shared tile store (`--tile-store`, default
`./tile_store/<row>_<col>/<year>/`). Per-year stitched mosaics for
this ROI are kept under `<output_dir>/stitched/` for inspection.

