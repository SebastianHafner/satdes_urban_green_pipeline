import geopandas as gpd
from shapely.geometry import Polygon

from urban_green.roi import rasterize_roi_mask, region_masks, roi_grid_transform, tile_roi


def test_tile_roi_covers_only_intersecting_tiles():
    # An L-shaped polygon: the fixed grid's candidate cells over its bounding
    # box include cells that never touch the shape, which tile_roi must
    # exclude (both those fully outside and those only touching the
    # concave corner's boundary with zero overlapping area).
    poly = Polygon([(0, 0), (20000, 0), (20000, 10000), (10000, 10000), (10000, 20000), (0, 20000)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:3006")
    tiles = tile_roi(gdf, tile_size=10000)
    assert {(t.row, t.col) for t in tiles} == {(0, 0), (0, 1), (1, 0)}


def test_tile_roi_uses_fixed_grid_independent_of_roi_bounds():
    # Tile row/col must come from absolute coordinates on the fixed grid,
    # not from the ROI's own bounding box.
    poly = Polygon([(12000, 22000), (18000, 22000), (18000, 28000), (12000, 28000)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:3006")
    tiles = tile_roi(gdf, tile_size=10000)
    assert {(t.row, t.col) for t in tiles} == {(2, 1)}  # x in [12000,18000) -> col 1; y in [22000,28000) -> row 2


def test_tile_roi_shared_border_tile_has_same_identity_across_rois():
    # Two adjacent ROIs sharing a border tile must resolve to the identical
    # (row, col, bounds) for that tile -- this is what makes tiles reusable
    # across different ROI runs via the tile store.
    poly_a = Polygon([(5000, 5000), (15000, 5000), (15000, 15000), (5000, 15000)])
    poly_b = Polygon([(15000, 5000), (25000, 5000), (25000, 15000), (15000, 15000)])
    gdf_a = gpd.GeoDataFrame({"geometry": [poly_a]}, crs="EPSG:3006")
    gdf_b = gpd.GeoDataFrame({"geometry": [poly_b]}, crs="EPSG:3006")

    tiles_a = {(t.row, t.col): t.bounds for t in tile_roi(gdf_a, tile_size=10000)}
    tiles_b = {(t.row, t.col): t.bounds for t in tile_roi(gdf_b, tile_size=10000)}

    shared = set(tiles_a) & set(tiles_b)
    assert shared, "expected at least one shared tile between adjacent ROIs"
    for key in shared:
        assert tiles_a[key] == tiles_b[key]


def test_tile_roi_dedupes_tiles_across_features():
    # Two ROI features covering the same fixed-grid tile must not produce
    # duplicate entries for it.
    poly_a = Polygon([(0, 0), (5000, 0), (5000, 5000), (0, 5000)])
    poly_b = Polygon([(5000, 5000), (9000, 5000), (9000, 9000), (5000, 9000)])
    gdf = gpd.GeoDataFrame({"geometry": [poly_a, poly_b]}, crs="EPSG:3006")
    tiles = tile_roi(gdf, tile_size=10000)
    assert {(t.row, t.col) for t in tiles} == {(0, 0)}
    assert len(tiles) == 1


def test_tile_roi_rejects_non_grid_crs():
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
    try:
        tile_roi(gdf, tile_size=10000)
        assert False, "expected ValueError for a non-EPSG:3006 ROI"
    except ValueError:
        pass


def test_rasterize_roi_mask_matches_grid_shape():
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:3006")
    transform, shape = roi_grid_transform(gdf, grid_size=10)
    mask = rasterize_roi_mask(gdf, transform, shape)
    assert mask.shape == shape
    assert mask.all()  # full 100x100 box, grid-aligned -> every cell covered


def test_region_masks_are_disjoint():
    gdf = gpd.GeoDataFrame(
        {
            "name": ["a", "b"],
            "geometry": [
                Polygon([(0, 0), (50, 0), (50, 100), (0, 100)]),
                Polygon([(50, 0), (100, 0), (100, 100), (50, 100)]),
            ],
        },
        crs="EPSG:3006",
    )
    transform, shape = roi_grid_transform(gdf, grid_size=10)
    masks = region_masks(gdf, transform, shape, "name")
    assert set(masks) == {"a", "b"}
    assert not (masks["a"] & masks["b"]).any()
