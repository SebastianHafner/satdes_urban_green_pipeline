"""Global-GMM + temporally-smoothed NDVI thresholding."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

# Plausible range for a vegetated/non-vegetated cut on annual max-NDVI. An automatically
# fitted threshold outside it usually means the NDVI distribution isn't cleanly bimodal.
REASONABLE_THRESHOLD_RANGE = (0.2, 0.6)

# Maximum number of pixels the GMM is fitted on. Larger ROIs are randomly subsampled: the
# threshold is stable at this size, and fitting tens of millions of float32 values makes
# the GMM numerically unstable (it can collapse into two identical components).
GMM_SAMPLE_SIZE_DEFAULT = 1_000_000


def fit_gmm(values: np.ndarray, n_components: int = 2, random_state: int = 42) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(values.reshape(-1, 1))
    return gmm


def gmm_threshold(gmm: GaussianMixture) -> float:
    """Decision threshold at the intersection of the two fitted Gaussians."""
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    order = np.argsort(means)
    m1, m2 = means[order]
    s1, s2 = stds[order]
    w1, w2 = weights[order]
    f = lambda x: w1 * norm.pdf(x, m1, s1) - w2 * norm.pdf(x, m2, s2)  # noqa: E731
    if not f(m1) > 0 > f(m2):
        raise ValueError(
            f"GMM did not separate two NDVI classes (component means {m1:.3f} and {m2:.3f}); "
            "no threshold between them. Inspect the NDVI distribution and set --ndvi-threshold manually."
        )
    return brentq(f, m1, m2)


class ValueSampler:
    """Uniform random sample (without replacement) of at most `sample_size` values, fed in chunks.

    Every value gets a random key and the `sample_size` values with the smallest keys are
    kept, so memory stays bounded however many values pass through. While no more than
    `sample_size` values have been added, all of them are kept.
    """

    def __init__(self, sample_size: int = GMM_SAMPLE_SIZE_DEFAULT, random_state: int = 42):
        self.sample_size = sample_size
        self.n_seen = 0
        self._rng = np.random.default_rng(random_state)
        self._keys = np.empty(0, dtype=np.float64)
        self._values = np.empty(0, dtype=np.float32)

    def add(self, values: np.ndarray) -> None:
        """Add values; non-finite ones (nodata) are ignored."""
        values = values[np.isfinite(values)].astype(np.float32)
        self.n_seen += values.size
        keys = np.concatenate([self._keys, self._rng.random(values.size)])
        values = np.concatenate([self._values, values])
        if values.size > self.sample_size:
            keep = np.argpartition(keys, self.sample_size - 1)[: self.sample_size]
            keys, values = keys[keep], values[keep]
        self._keys, self._values = keys, values

    @property
    def sample(self) -> np.ndarray:
        return self._values


def fit_sampled_threshold(sampler: ValueSampler) -> float:
    """Fit the GMM threshold on a ValueSampler's sample."""
    if sampler.n_seen < 2:
        raise ValueError("Not enough valid NDVI pixels inside the ROI to fit a threshold.")
    if sampler.n_seen > sampler.sample_size:
        logger.info("Fitting GMM on %d of %d valid pixels (random sample)", sampler.sample.size, sampler.n_seen)
    else:
        logger.info("Fitting GMM on all %d valid pixels", sampler.n_seen)
    gmm = fit_gmm(sampler.sample.astype(np.float64))
    threshold = gmm_threshold(gmm)
    logger.info("Fitted NDVI threshold: %.4f", threshold)
    return threshold


def fit_global_threshold(
    ndvi_stack: np.ndarray, mask: np.ndarray, sample_size: int = GMM_SAMPLE_SIZE_DEFAULT, random_state: int = 42
) -> float:
    """Fit one GMM on urban-masked NDVI values pooled across all years.

    ndvi_stack: (T, H, W) float array. mask: (H, W) bool array, True = inside ROI.
    If there are more than `sample_size` valid pixels, the GMM is fitted on a random
    sample of `sample_size` of them (reproducible via `random_state`).
    """
    sampler = ValueSampler(sample_size, random_state)
    sampler.add(ndvi_stack[:, mask])
    return fit_sampled_threshold(sampler)


def classify(ndvi_stack: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    """Binary vegetated (1) / non-vegetated (0) classification; NaN outside the ROI or nodata."""
    binary = (ndvi_stack >= threshold).astype(np.float32)
    binary[:, ~mask] = np.nan
    binary[np.isnan(ndvi_stack)] = np.nan
    return binary


def majority_vote_smooth(binary_stack: np.ndarray, window: int = 3) -> np.ndarray:
    """Symmetric sliding-window majority vote along the year axis (axis 0).
    Edge years are padded by repeating the first/last year.
    """
    if window % 2 == 0:
        raise ValueError("majority_vote window must be odd for a symmetric vote.")
    half = window // 2
    padded = np.concatenate(
        [np.repeat(binary_stack[[0]], half, axis=0), binary_stack, np.repeat(binary_stack[[-1]], half, axis=0)],
        axis=0,
    )
    smoothed = np.full_like(binary_stack, np.nan)
    for i in range(binary_stack.shape[0]):
        win = padded[i : i + window]
        # Pixels that are NaN in every year of the window (outside the ROI) have nothing to
        # average; their result is overwritten with NaN below, so the warning is noise.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            majority = np.nanmean(win, axis=0) >= 0.5
        layer = majority.astype(np.float32)
        layer[np.isnan(binary_stack[i])] = np.nan
        smoothed[i] = layer
    return smoothed


@dataclass
class ThresholdResult:
    threshold: float
    binary_stack: np.ndarray  # (T, H, W), pre-smoothing
    smoothed_stack: np.ndarray  # (T, H, W), post majority-vote


def check_threshold_range(threshold: float) -> None:
    """Warn if an automatically fitted threshold falls outside REASONABLE_THRESHOLD_RANGE."""
    low, high = REASONABLE_THRESHOLD_RANGE
    if not low <= threshold <= high:
        logger.warning(
            "Automatically fitted NDVI threshold %.3f is outside the expected range [%.2f, %.2f]; "
            "the NDVI distribution of this ROI may not be bimodal. Inspect figures/ndvi_distribution.png "
            "and consider setting a threshold manually.",
            threshold,
            low,
            high,
        )


def run_thresholding(
    ndvi_stack: np.ndarray,
    mask: np.ndarray,
    window: int = 3,
    threshold: Optional[float] = None,
    gmm_sample_size: int = GMM_SAMPLE_SIZE_DEFAULT,
) -> ThresholdResult:
    """Classify and smooth the NDVI stack; fits a global GMM threshold unless `threshold` is given."""
    if threshold is None:
        threshold = fit_global_threshold(ndvi_stack, mask, sample_size=gmm_sample_size)
        check_threshold_range(threshold)
    binary_stack = classify(ndvi_stack, mask, threshold)
    smoothed_stack = majority_vote_smooth(binary_stack, window=window)
    return ThresholdResult(threshold=threshold, binary_stack=binary_stack, smoothed_stack=smoothed_stack)
