"""Shared, cross-run cache of per-tile annual max-NDVI acquisitions.

Backed by SQLite (stdlib, no new dependency). A row is only written after a
tile has been successfully acquired and written to disk, so a crash or
interrupted run never leaves a tile falsely marked as available.

The cache key is (tile_row, tile_col, year) only -- NOT crs/cloud_threshold.
This maximizes reuse across runs, but means changing --crs or
--cloud-threshold against an existing store will silently reuse tiles
acquired under the old parameters. Use --force-reacquire (or point at a
fresh --tile-store) if you change either of those.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    tile_row INTEGER NOT NULL,
    tile_col INTEGER NOT NULL,
    year INTEGER NOT NULL,
    crs TEXT NOT NULL,
    cloud_threshold INTEGER NOT NULL,
    ndvi_path TEXT NOT NULL,
    obs_path TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    PRIMARY KEY (tile_row, tile_col, year)
);
"""


@dataclasses.dataclass
class TileRecord:
    row: int
    col: int
    year: int
    crs: str
    cloud_threshold: int
    ndvi_path: Path
    obs_path: Path
    processed_at: str


class TileStore:
    """Filesystem + SQLite catalog of acquired (row, col, year) NDVI tiles."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.root / "catalog.sqlite")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TileStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def tile_dir(self, row: int, col: int, year: int) -> Path:
        return self.root / f"{row}_{col}" / str(year)

    def paths(self, row: int, col: int, year: int) -> tuple:
        tile_dir = self.tile_dir(row, col, year)
        return tile_dir / "ndvi.tif", tile_dir / "obs.tif"

    def is_available(self, row: int, col: int, year: int) -> Optional[TileRecord]:
        """Return the cached TileRecord if this (row, col, year) has already been
        acquired AND its files still exist on disk; None otherwise."""
        cur = self._conn.execute(
            "SELECT tile_row, tile_col, year, crs, cloud_threshold, ndvi_path, obs_path, processed_at "
            "FROM tiles WHERE tile_row = ? AND tile_col = ? AND year = ?",
            (row, col, year),
        )
        row_data = cur.fetchone()
        if row_data is None:
            return None
        record = TileRecord(
            row=row_data[0],
            col=row_data[1],
            year=row_data[2],
            crs=row_data[3],
            cloud_threshold=row_data[4],
            ndvi_path=Path(row_data[5]),
            obs_path=Path(row_data[6]),
            processed_at=row_data[7],
        )
        if not (record.ndvi_path.exists() and record.obs_path.exists()):
            return None  # catalog entry without files on disk (e.g. manually deleted) -- treat as missing
        return record

    def register(
        self, row: int, col: int, year: int, crs: str, cloud_threshold: int, ndvi_path: Path, obs_path: Path
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tiles "
            "(tile_row, tile_col, year, crs, cloud_threshold, ndvi_path, obs_path, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row, col, year, crs, cloud_threshold, str(ndvi_path), str(obs_path), datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
