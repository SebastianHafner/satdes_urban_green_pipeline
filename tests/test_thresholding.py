import numpy as np

from urban_green import thresholding


def _synthetic_stack(n_years=5, size=20, seed=0):
    rng = np.random.default_rng(seed)
    veg = rng.normal(0.75, 0.05, size=(n_years, size, size))
    nonveg = rng.normal(0.15, 0.05, size=(n_years, size, size))
    label = rng.integers(0, 2, size=(size, size))
    stack = np.where(label[None] == 1, veg, nonveg).astype(np.float32)
    mask = np.ones((size, size), dtype=bool)
    return stack, mask


def test_gmm_threshold_separates_two_clusters():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0.1, 0.02, 500), rng.normal(0.8, 0.02, 500)])
    gmm = thresholding.fit_gmm(values)
    threshold = thresholding.gmm_threshold(gmm)
    assert 0.1 < threshold < 0.8


def test_run_thresholding_end_to_end():
    stack, mask = _synthetic_stack()
    result = thresholding.run_thresholding(stack, mask, window=3)
    assert 0.3 < result.threshold < 0.6
    assert result.smoothed_stack.shape == stack.shape
    finite = result.smoothed_stack[np.isfinite(result.smoothed_stack)]
    assert set(np.unique(finite)).issubset({0.0, 1.0})


def test_mask_excludes_pixels_outside_roi():
    stack, mask = _synthetic_stack(size=10)
    mask[:, :5] = False  # left half outside the ROI
    result = thresholding.run_thresholding(stack, mask, window=3)
    assert np.all(np.isnan(result.smoothed_stack[:, :, :5]))
    assert not np.all(np.isnan(result.smoothed_stack[:, :, 5:]))


def test_majority_vote_smooths_single_year_flip():
    # A pixel that is green every year except a single spurious flip in year 2
    # should be smoothed back to green by the 3-year majority vote.
    binary = np.ones((5, 1, 1), dtype=np.float32)
    binary[2, 0, 0] = 0.0
    smoothed = thresholding.majority_vote_smooth(binary, window=3)
    assert smoothed[2, 0, 0] == 1.0


def test_majority_vote_rejects_even_window():
    binary = np.ones((3, 1, 1), dtype=np.float32)
    try:
        thresholding.majority_vote_smooth(binary, window=2)
        assert False, "expected ValueError for even window"
    except ValueError:
        pass
