# -*- coding = utf-8 -*-
# @Time: 2025/2/18 19:12
# @Author: wisehone
# @File: threshold.py
# @SoftWare: PyCharm

import numpy as np

THRESHOLD_METHODS = (
    'spot',                 # original SPOT / POT threshold
    'quantile_test',        # top q% of the test-set scores
    'quantile_train',       # top q% of the train-set scores
    'quantile_train_test',  # top q% of the concatenated train+test scores
)


def _check_quantile(quantile):
    if not 0 < quantile < 100:
        raise ValueError(f'quantile must be in (0, 100), got {quantile}')


def _quantile_threshold(scores, quantile):
    _check_quantile(quantile)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError('cannot compute a quantile threshold from empty scores')
    return float(np.percentile(scores, 100.0 - quantile))


def select_threshold(method, train_scores, test_scores, quantile=4.0,
                     spot_level=0.02, spot_scale=3.0):
    """Select an anomaly-score threshold with the requested strategy.

    Args:
        method: one of :data:`THRESHOLD_METHODS`.
        train_scores: anomaly scores computed on the training data.
        test_scores: anomaly scores computed on the test data.
        quantile: tail percentage used by the quantile-based methods.
        spot_level: initial SPOT quantile (only used by ``spot``).
        spot_scale: SPOT threshold scale factor (only used by ``spot``).

    Returns:
        float: the selected threshold.
    """
    if method not in THRESHOLD_METHODS:
        raise ValueError(f'unknown threshold method {method!r}, '
                         f'expected one of {THRESHOLD_METHODS}')

    if method == 'spot':
        # imported lazily so the quantile strategies do not require SPOT/sklearn
        from utils.pot import POTThresholdDetector
        detector = POTThresholdDetector(level=spot_level, scale_factor=spot_scale)
        detector.fit(train_scores, test_scores)
        return float(detector.threshold)

    if method == 'quantile_test':
        return _quantile_threshold(test_scores, quantile)

    if method == 'quantile_train':
        return _quantile_threshold(train_scores, quantile)

    # quantile_train_test
    combined = np.concatenate([
        np.asarray(train_scores, dtype=np.float64).reshape(-1),
        np.asarray(test_scores, dtype=np.float64).reshape(-1),
    ])
    return _quantile_threshold(combined, quantile)
