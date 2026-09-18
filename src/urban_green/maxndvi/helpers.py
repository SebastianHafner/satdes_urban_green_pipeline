"""Vendored (trimmed) from satdes_maxndvi/data_loading/helpers.py.

Only the pieces needed to acquire an annual max-NDVI + observation-count
composite from Digital Earth Sweden are kept.
"""
from __future__ import annotations

import io

import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from shapely.geometry.base import BaseGeometry


def get_start_date(year: int) -> str:
    return f"{year}-06-01"


def get_end_date(year: int) -> str:
    return f"{year}-09-01"


def get_bbox(geom: BaseGeometry, crs: str) -> dict:
    minx, miny, maxx, maxy = geom.bounds
    return {"west": minx, "south": miny, "east": maxx, "north": maxy, "crs": crs}


def run_job(datacube, single_image: bool = True, max_files: int = 200):
    if single_image:
        job = datacube.create_job(out_format="gtiff")
    else:
        job = datacube.create_job(out_format="NetCDF", options={"max_files": max_files})
    job.start_and_wait()
    return job.get_results()


def extract_image(result) -> xr.DataArray:
    """Download the single GeoTIFF asset of a job result as a (band, y, x) DataArray."""
    image_tiff = None
    for asset in result.get_assets():
        if asset.metadata["type"] == "image/tiff; application=geotiff":
            image_tiff = asset.load_bytes()
    if image_tiff is None:
        raise RuntimeError("No GeoTIFF asset found in job result.")
    image_dataset = xr.open_dataset(io.BytesIO(image_tiff), engine="rasterio")
    image_xarray = image_dataset.to_array().squeeze("variable")
    return image_xarray.astype("float32")
