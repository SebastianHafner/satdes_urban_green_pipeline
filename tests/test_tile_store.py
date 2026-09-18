from urban_green.tile_store import TileStore


def _write(path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_is_available_none_when_not_registered(tmp_path):
    store = TileStore(tmp_path / "store")
    assert store.is_available(0, 0, 2020) is None


def test_register_then_is_available_round_trip(tmp_path):
    store = TileStore(tmp_path / "store")
    ndvi_path, obs_path = store.paths(3, -2, 2021)
    _write(ndvi_path)
    _write(obs_path)

    store.register(3, -2, 2021, "EPSG:3006", 70, ndvi_path, obs_path)

    record = store.is_available(3, -2, 2021)
    assert record is not None
    assert record.row == 3
    assert record.col == -2
    assert record.year == 2021
    assert record.ndvi_path == ndvi_path
    assert record.obs_path == obs_path


def test_is_available_none_when_files_missing_on_disk(tmp_path):
    # A catalog row whose files were deleted out-of-band (e.g. manually)
    # must be treated as not available, not as a stale hit.
    store = TileStore(tmp_path / "store")
    ndvi_path, obs_path = store.paths(0, 0, 2020)
    _write(ndvi_path)
    _write(obs_path)
    store.register(0, 0, 2020, "EPSG:3006", 70, ndvi_path, obs_path)

    ndvi_path.unlink()

    assert store.is_available(0, 0, 2020) is None


def test_register_is_idempotent_overwrite(tmp_path):
    store = TileStore(tmp_path / "store")
    ndvi_path, obs_path = store.paths(0, 0, 2020)
    _write(ndvi_path)
    _write(obs_path)

    store.register(0, 0, 2020, "EPSG:3006", 70, ndvi_path, obs_path)
    store.register(0, 0, 2020, "EPSG:3006", 50, ndvi_path, obs_path)  # different cloud_threshold, same key

    record = store.is_available(0, 0, 2020)
    assert record.cloud_threshold == 50


def test_store_persists_across_reopen(tmp_path):
    store_dir = tmp_path / "store"
    store = TileStore(store_dir)
    ndvi_path, obs_path = store.paths(1, 1, 2019)
    _write(ndvi_path)
    _write(obs_path)
    store.register(1, 1, 2019, "EPSG:3006", 70, ndvi_path, obs_path)
    store.close()

    reopened = TileStore(store_dir)
    assert reopened.is_available(1, 1, 2019) is not None
