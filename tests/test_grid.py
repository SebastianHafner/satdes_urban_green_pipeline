import shapely.geometry

from urban_green.grid import Tile, national_grid_tiles


def test_tile_name_and_geometry():
    tile = Tile(row=2, col=-1, bounds=(-10000.0, 20000.0, 0.0, 30000.0))
    assert tile.name == "2_-1"
    assert tile.geometry.equals(shapely.geometry.box(-10000.0, 20000.0, 0.0, 30000.0))


def test_national_grid_tiles_single_cell():
    footprint = shapely.geometry.box(100.0, 100.0, 200.0, 200.0)
    tiles = national_grid_tiles(footprint, tile_size=10_000.0)
    assert len(tiles) == 1
    assert (tiles[0].row, tiles[0].col) == (0, 0)
    assert tiles[0].bounds == (0.0, 0.0, 10_000.0, 10_000.0)


def test_national_grid_tiles_excludes_boundary_only_touch():
    # A footprint whose right edge sits exactly on a tile boundary must not
    # pull in the neighboring tile, which would have zero overlapping area.
    footprint = shapely.geometry.box(0.0, 0.0, 10_000.0, 10_000.0)
    tiles = national_grid_tiles(footprint, tile_size=10_000.0)
    assert {(t.row, t.col) for t in tiles} == {(0, 0)}


def test_national_grid_tiles_negative_coordinates():
    footprint = shapely.geometry.box(-15_000.0, -5_000.0, -12_000.0, -1_000.0)
    tiles = national_grid_tiles(footprint, tile_size=10_000.0)
    assert {(t.row, t.col) for t in tiles} == {(-1, -2)}
