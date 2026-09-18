import numpy as np

from urban_green import statistics


def test_green_area_stats_change_column():
    stack = np.zeros((3, 4, 4), dtype=np.float32)
    stack[0] = 1  # all green year 1
    stack[1] = 0  # none green year 2
    stack[2] = 1  # all green year 3
    stats = statistics.green_area_stats(stack, [2018, 2019, 2020])

    assert stats.loc[0, "change_km2"] == 0
    assert stats.loc[1, "change_km2"] < 0
    assert stats.loc[2, "change_km2"] > 0
    assert stats.loc[0, "pct_cover"] == 100
    assert stats.loc[1, "pct_cover"] == 0


def test_green_area_stats_by_region():
    stack = np.ones((2, 4, 4), dtype=np.float32)
    masks = {
        "a": np.array([[True] * 4] * 2 + [[False] * 4] * 2),
        "b": np.array([[False] * 4] * 2 + [[True] * 4] * 2),
    }
    df = statistics.green_area_stats_by_region(stack, [2018, 2019], masks)
    assert set(df["region"]) == {"a", "b"}
    assert len(df) == 4


def test_change_summary_areas():
    loss_year = np.array([[2019, 0], [0, 2020]], dtype=np.uint16)
    gain_year = np.array([[0, 2020], [0, 0]], dtype=np.uint16)
    df = statistics.change_summary(loss_year, gain_year, [2018, 2019, 2020])
    row_2019 = df[df["year"] == 2019].iloc[0]
    row_2020 = df[df["year"] == 2020].iloc[0]
    assert row_2019["loss_km2"] == 1 * 100 / 1e6
    assert row_2020["loss_km2"] == 1 * 100 / 1e6
    assert row_2020["gain_km2"] == 1 * 100 / 1e6
