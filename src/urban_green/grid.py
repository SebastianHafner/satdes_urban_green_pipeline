"""Fixed national tiling grid, anchored at (0, 0) in EPSG:3006 (SWEREF99 TM).

Unlike a per-ROI bounding-box grid, a fixed-origin grid gives every tile a
CRS-derived identity (row, col) independent of which ROI requested it, so
the same tile can be acquired once and reused across different ROI runs
(see urban_green.tile_store). This mirrors the convention used by
Lantmateriet's national SWEREF99 TM index grid.
"""
from __future__ import annotations

import dataclasses
import math
from typing import List

import shapely.geometry
from shapely.geometry.base import BaseGeometry

GRID_CRS = "EPSG:3006"
TILE_SIZE_M_DEFAULT = 10_000.0


@dataclasses.dataclass(frozen=True)
class Tile:
    row: int
    col: int
    bounds: tuple  # (minx, miny, maxx, maxy), EPSG:3006

    @property
    def name(self) -> str:
        return f"{self.row}_{self.col}"

    @property
    def geometry(self) -> shapely.geometry.Polygon:
        return shapely.geometry.box(*self.bounds)


def national_grid_tiles(footprint: BaseGeometry, tile_size: float = TILE_SIZE_M_DEFAULT) -> List[Tile]:
    """Fixed grid tiles (origin (0, 0), EPSG:3006) that positively overlap `footprint`.

    `footprint` must already be in EPSG:3006. A tile is kept only if its
    intersection with the footprint has positive area -- a tile that merely
    touches the footprint's boundary (zero overlapping area) is excluded,
    since it has nothing to acquire.
    """
    fminx, fminy, fmaxx, fmaxy = footprint.bounds
    col_min = math.floor(fminx / tile_size)
    col_max = math.floor(fmaxx / tile_size)
    row_min = math.floor(fminy / tile_size)
    row_max = math.floor(fmaxy / tile_size)

    tiles = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            minx, miny = col * tile_size, row * tile_size
            maxx, maxy = minx + tile_size, miny + tile_size
            tile_box = shapely.geometry.box(minx, miny, maxx, maxy)
            if tile_box.intersection(footprint).area > 0:
                tiles.append(Tile(row=row, col=col, bounds=(minx, miny, maxx, maxy)))
    return tiles
