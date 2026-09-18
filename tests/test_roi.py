import geopandas as gpd
from shapely.geometry import Polygon

from urban_green.roi import rasterize_roi_mask, region_masks, roi_grid_transform, tile_roi


def test_tile_roi_covers_only_intersecting_tiles():
    # An L-shaped polygon: its bounding-box grid would include tiles that
    # never touch the shape, which tile_roi must exclude.
    poly = Polygon([(0, 0), (20000, 0), (20000, 10000), (10000, 10000), (10000, 20000), (0, 20000)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:3006")
    tiles = tile_roi(gdf, tile_size=10000)
    assert len(tiles) < 4  # full 2x2 bbox grid would be 4 tiles; the L-shape excludes one quadrant
    assert all(tiles.geometry.intersects(poly))


def test_tile_roi_uses_region_id_field():
    gdf = gpd.GeoDataFrame(
        {
            "name": ["north", "south"],
            "geometry": [
                Polygon([(0, 10000), (10000, 10000), (10000, 20000), (0, 20000)]),
                Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)]),
            ],
        },
        crs="EPSG:3006",
    )
    tiles = tile_roi(gdf, tile_size=10000, region_id_field="name")
    assert set(tiles["name"]) == {"north", "south"}


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
