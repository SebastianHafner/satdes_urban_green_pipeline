"""Tile-based access to per-tile annual max-NDVI rasters on one common working grid.

Tiles come from the fixed national grid and are resolved through the shared
urban_green.tile_store.TileStore, so only the tiles relevant to the current
ROI are used even when the store holds many more tiles overall. Instead of
mosaicking every year into memory, the pipeline processes one tile (all
years) at a time, writing each into its window of the working grid. The
per-year "stitched" mosaics are written as VRTs that reference
the tile files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds

from urban_green.grid import Tile
from urban_green.tile_store import TileStore

# tile -> {year: ndvi raster path}
TilePaths = Dict[Tile, Dict[int, Path]]


def resolve_tile_paths(tile_store: TileStore, tiles: List[Tile], years: List[int]) -> TilePaths:
    """Look up every (tile, year) NDVI raster in the store; raise if any is missing."""
    tile_paths = {}
    for tile in tiles:
        tile_paths[tile] = {}
        for year in years:
            record = tile_store.is_available(tile.row, tile.col, year)
            if record is None:
                raise FileNotFoundError(
                    f"Tile ({tile.row}, {tile.col}) for year {year} is not in the tile store "
                    f"at {tile_store.root} -- run acquisition before processing."
                )
            tile_paths[tile][year] = record.ndvi_path
    return tile_paths


def tile_grid(tile_files: List[Path]) -> Tuple[Affine, Tuple[int, int]]:
    """Working grid covering the union of the given tile rasters (as rasterio.merge would mosaic them).

    Pixel size and alignment come from the first raster; tile_window() checks that
    every tile actually sits on this grid.
    """
    bounds = []
    for path in tile_files:
        with rasterio.open(path) as src:
            bounds.append(src.bounds)
            if len(bounds) == 1:
                res_x, res_y = src.res
    minx = min(b.left for b in bounds)
    miny = min(b.bottom for b in bounds)
    maxx = max(b.right for b in bounds)
    maxy = max(b.top for b in bounds)
    width = int(round((maxx - minx) / res_x))
    height = int(round((maxy - miny) / res_y))
    return Affine(res_x, 0.0, minx, 0.0, -res_y, maxy), (height, width)


def tile_window(src, transform: Affine) -> Window:
    """Integer pixel window of an open tile raster within the working grid; raises if not grid-aligned."""
    if not np.allclose(src.res, (transform.a, -transform.e), atol=1e-6):
        raise ValueError(f"Tile {src.name} has pixel size {src.res}, but the working grid uses {transform.a}.")
    window = from_bounds(*src.bounds, transform=transform)
    offsets = (window.col_off, window.row_off, window.width, window.height)
    if not all(abs(v - round(v)) < 1e-6 for v in offsets):
        raise ValueError(f"Tile {src.name} is not aligned with the working pixel grid (window {window}).")
    return Window(*(int(round(v)) for v in offsets))


def read_tile_stack(year_paths: Dict[int, Path], years: List[int], transform: Affine) -> Tuple[np.ndarray, Window]:
    """Read one tile for all years as a (T, h, w) float32 stack, plus its window in the working grid.

    Every year must share the first year's pixel grid exactly -- years are stacked
    directly, never resampled.
    """
    stack = None
    for i, year in enumerate(years):
        with rasterio.open(year_paths[year]) as src:
            if stack is None:
                window = tile_window(src, transform)
                first_grid = (src.height, src.width, tuple(src.transform))
                stack = np.full((len(years), src.height, src.width), np.nan, dtype=np.float32)
            elif not (
                (src.height, src.width) == first_grid[:2]
                and np.allclose(tuple(src.transform), first_grid[2], atol=1e-6)
            ):
                raise ValueError(
                    f"Tile {year_paths[year]} for year {year} has a different grid than year {years[0]} "
                    f"(({src.height}, {src.width}), {src.transform} vs {first_grid[:2]}, {first_grid[2]}) -- "
                    "DES returned an inconsistent pixel grid across years for the same tile."
                )
            stack[i] = src.read(1).astype(np.float32)
    return stack, window


def write_year_vrt(tile_files: List[Path], transform: Affine, out_shape: Tuple[int, int], crs: str, out_file: Path) -> Path:
    """Write a VRT mosaic of the given tile rasters on the working grid, referencing (not copying) them.

    Tile paths are stored as absolute paths, so the VRT breaks if the tile store is moved.
    """
    height, width = out_shape
    sources = []
    for path in tile_files:
        with rasterio.open(path) as src:
            window = tile_window(src, transform)
            dtype = src.dtypes[0]
            nodata = src.nodata
            sources.append(
                f"""    <ComplexSource>
      <SourceFilename relativeToVRT="0">{escape(str(Path(path).resolve()))}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}" />
      <DstRect xOff="{window.col_off}" yOff="{window.row_off}" xSize="{window.width}" ySize="{window.height}" />
    </ComplexSource>"""
            )
    gdal_dtype = {"float16": "Float32", "float32": "Float32", "float64": "Float64", "uint8": "Byte"}[dtype]
    nodata_xml = "" if nodata is None else f"\n    <NoDataValue>{'nan' if np.isnan(nodata) else nodata}</NoDataValue>"
    geotransform = ", ".join(repr(float(v)) for v in transform.to_gdal())
    srs = escape(rasterio.crs.CRS.from_user_input(crs).to_wkt())
    sources_xml = "\n".join(sources)
    out_file.write_text(
        f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>{srs}</SRS>
  <GeoTransform>{geotransform}</GeoTransform>
  <VRTRasterBand dataType="{gdal_dtype}" band="1">{nodata_xml}
{sources_xml}
  </VRTRasterBand>
</VRTDataset>
"""
    )
    return out_file


def write_year_vrts(
    tile_paths: TilePaths, years: List[int], transform: Affine, out_shape: Tuple[int, int], crs: str, out_dir: Path
) -> Dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        year: write_year_vrt(
            [paths[year] for paths in tile_paths.values()], transform, out_shape, crs, out_dir / f"maxndvi_{year}.vrt"
        )
        for year in years
    }
