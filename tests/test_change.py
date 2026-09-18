import numpy as np

from urban_green import change


def test_compute_change_years_detects_first_transition():
    veg_stack = np.zeros((4, 3, 3), dtype=np.float32)
    veg_stack[2:, 0, 0] = 1  # gain at year index 2
    veg_stack[:2, 1, 1] = 1
    veg_stack[2:, 1, 1] = 0  # loss at year index 2
    years = [2018, 2019, 2020, 2021]

    result = change.compute_change_years(veg_stack, years)
    assert result["gain_year"][0, 0] == 2020
    assert result["loss_year"][1, 1] == 2020
    assert result["gain_year"][2, 2] == 0  # no change -> fill value
    assert result["loss_year"][2, 2] == 0


def test_compute_change_years_keeps_first_of_multiple_transitions():
    # gain in year index 1, loss in year index 2, gain again in year index 3:
    # the FIRST gain (year index 1) must be kept, not the second.
    veg_stack = np.array([0, 1, 0, 1], dtype=np.float32).reshape(4, 1, 1)
    years = [2018, 2019, 2020, 2021]
    result = change.compute_change_years(veg_stack, years)
    assert result["gain_year"][0, 0] == 2019
    assert result["loss_year"][0, 0] == 2020


def test_compute_change_years_ignores_nan_transitions():
    veg_stack = np.array([1, np.nan, 0], dtype=np.float32).reshape(3, 1, 1)
    years = [2018, 2019, 2020]
    result = change.compute_change_years(veg_stack, years)
    assert result["loss_year"][0, 0] == 0
    assert result["gain_year"][0, 0] == 0
