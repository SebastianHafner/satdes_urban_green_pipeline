from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DES_ENDPOINT_DEFAULT = "https://openeo.digitalearth.se"
# Placeholder test credentials from satdes_maxndvi, used only as a fallback so the
# demonstrator still runs out of the box. Set DES_USERNAME / DES_PASSWORD for real access.
DES_USERNAME_DEFAULT = "testuser"
DES_PASSWORD_DEFAULT = "secretpassword"


@dataclass
class PipelineConfig:
    """Configuration for a single multi-year urban green statistics pipeline run."""

    roi_path: Path
    years: List[int]
    output_dir: Path

    tile_size: int = 10_000  # meters, per-tile DES request size
    cloud_threshold: int = 70  # max eo:cloud_cover percentage per Sentinel-2 scene
    crs: str = "EPSG:3006"  # SWEREF99 TM
    grid_size: int = 10  # NDVI pixel size in meters
    majority_vote_window: int = 3  # years, for temporal smoothing of the vegetation classification

    # Optional column in the ROI file used to name/group sub-regions (e.g. municipality name).
    # When set, the ROI is tiled and statistics are reported per named feature.
    region_id_field: Optional[str] = None

    des_endpoint: str = DES_ENDPOINT_DEFAULT

    @property
    def des_username(self) -> str:
        return os.environ.get("DES_USERNAME", DES_USERNAME_DEFAULT)

    @property
    def des_password(self) -> str:
        return os.environ.get("DES_PASSWORD", DES_PASSWORD_DEFAULT)

    def __post_init__(self) -> None:
        self.roi_path = Path(self.roi_path)
        self.output_dir = Path(self.output_dir)
        self.years = sorted(self.years)
        if len(self.years) < 2:
            raise ValueError("At least two years are required to compute year-on-year statistics.")
