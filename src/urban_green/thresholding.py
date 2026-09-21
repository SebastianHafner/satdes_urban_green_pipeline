"""Global-GMM + temporally-smoothed NDVI thresholding."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


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
    return brentq(f, m1, m2)


def fit_global_threshold(ndvi_stack: np.ndarray, mask: np.ndarray) -> float:
    """Fit one GMM on urban-masked NDVI values pooled across all years.

    ndvi_stack: (T, H, W) float array. mask: (H, W) bool array, True = inside ROI.
    """
    values = ndvi_stack[:, mask]
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("Not enough valid NDVI pixels inside the ROI to fit a threshold.")
    gmm = fit_gmm(values)
    return gmm_threshold(gmm)


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
        with np.errstate(invalid="ignore"):
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


def run_thresholding(ndvi_stack: np.ndarray, mask: np.ndarray, window: int = 3) -> ThresholdResult:
    threshold = fit_global_threshold(ndvi_stack, mask)
    binary_stack = classify(ndvi_stack, mask, threshold)
    smoothed_stack = majority_vote_smooth(binary_stack, window=window)
    return ThresholdResult(threshold=threshold, binary_stack=binary_stack, smoothed_stack=smoothed_stack)
