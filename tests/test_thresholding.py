import numpy as np
import pytest

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


def test_user_threshold_skips_automatic_detection(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("automatic threshold detection should not run")

    monkeypatch.setattr(thresholding, "fit_global_threshold", fail)
    stack, mask = _synthetic_stack()
    result = thresholding.run_thresholding(stack, mask, window=3, threshold=0.5)
    assert result.threshold == 0.5
    expected = (stack >= 0.5).astype(np.float32)
    np.testing.assert_array_equal(result.binary_stack, expected)


def test_out_of_range_fitted_threshold_warns(monkeypatch, caplog):
    monkeypatch.setattr(thresholding, "fit_global_threshold", lambda *args, **kwargs: 0.75)
    stack, mask = _synthetic_stack()
    with caplog.at_level("WARNING", logger="urban_green.thresholding"):
        thresholding.run_thresholding(stack, mask, window=3)
    assert "outside the expected range" in caplog.text


def test_in_range_fitted_threshold_does_not_warn(caplog):
    stack, mask = _synthetic_stack()
    with caplog.at_level("WARNING", logger="urban_green.thresholding"):
        result = thresholding.run_thresholding(stack, mask, window=3)
    assert 0.3 < result.threshold < 0.6
    assert "outside the expected range" not in caplog.text


def test_user_threshold_out_of_range_does_not_warn(caplog):
    stack, mask = _synthetic_stack()
    with caplog.at_level("WARNING", logger="urban_green.thresholding"):
        thresholding.run_thresholding(stack, mask, window=3, threshold=0.8)
    assert caplog.text == ""


def _record_fit_gmm_sizes(monkeypatch):
    sizes = []
    real_fit_gmm = thresholding.fit_gmm

    def recording_fit_gmm(values, *args, **kwargs):
        sizes.append(values.size)
        return real_fit_gmm(values, *args, **kwargs)

    monkeypatch.setattr(thresholding, "fit_gmm", recording_fit_gmm)
    return sizes


def test_gmm_fit_samples_large_roi(monkeypatch):
    sizes = _record_fit_gmm_sizes(monkeypatch)
    stack, mask = _synthetic_stack()  # 5 * 20 * 20 = 2000 valid pixels
    threshold = thresholding.fit_global_threshold(stack, mask, sample_size=500)
    assert sizes == [500]
    assert 0.3 < threshold < 0.6


def test_gmm_fit_uses_all_pixels_of_small_roi(monkeypatch):
    sizes = _record_fit_gmm_sizes(monkeypatch)
    stack, mask = _synthetic_stack()
    thresholding.fit_global_threshold(stack, mask, sample_size=5000)
    assert sizes == [stack.size]


def test_gmm_threshold_rejects_identical_components():
    gmm = thresholding.GaussianMixture(n_components=2)
    gmm.means_ = np.array([[1.08], [1.08]])
    gmm.covariances_ = np.array([[[0.49]], [[0.49]]])
    gmm.weights_ = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="did not separate two NDVI classes"):
        thresholding.gmm_threshold(gmm)


def test_value_sampler_keeps_everything_below_sample_size():
    sampler = thresholding.ValueSampler(sample_size=10)
    sampler.add(np.array([0.1, np.nan, 0.2], dtype=np.float32))
    sampler.add(np.array([0.3], dtype=np.float32))
    assert sampler.n_seen == 3
    np.testing.assert_array_equal(np.sort(sampler.sample), np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_value_sampler_caps_sample_and_is_reproducible():
    values = np.arange(10_000, dtype=np.float32)
    samples = []
    for _ in range(2):
        sampler = thresholding.ValueSampler(sample_size=1000, random_state=7)
        for chunk in np.array_split(values, 13):
            sampler.add(chunk)
        samples.append(np.sort(sampler.sample))
    assert sampler.n_seen == values.size
    assert samples[0].size == 1000
    assert np.unique(samples[0]).size == 1000  # without replacement
    np.testing.assert_array_equal(samples[0], samples[1])


def test_value_sampler_is_uniform_across_chunks():
    # Early and late chunks must be equally represented, not biased by arrival order.
    sampler = thresholding.ValueSampler(sample_size=20_000, random_state=0)
    for chunk in np.array_split(np.arange(200_000, dtype=np.float32), 50):
        sampler.add(chunk)
    first_half_share = np.mean(sampler.sample < 100_000)
    assert 0.48 < first_half_share < 0.52
