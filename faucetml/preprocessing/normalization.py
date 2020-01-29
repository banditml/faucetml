"""
Code taken from Facebook ReAgent and modified for Faucet ML.

Original copyright:
    Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

Original BSD License:
    https://github.com/facebookresearch/ReAgent/blob/master/LICENSE
"""

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import six
import torch
from scipy import stats
from scipy.stats.mstats import mquantiles

from . import identify_types
from ..utils import get_logger


logger = get_logger(__name__)


MINIMUM_SAMPLES_TO_IDENTIFY = 750
BOX_COX_MAX_STDDEV = 1e8
BOX_COX_MARGIN = 1e-4
MISSING_VALUE = -1337.1337
MAX_FEATURE_VALUE = 6.0
MIN_FEATURE_VALUE = MAX_FEATURE_VALUE * -1
EPS = 1e-6


@dataclass
class NormalizationParameters:
    feature_type: str
    boxcox_lambda: Optional[float] = None
    boxcox_shift: Optional[float] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    mode: Optional[float] = None
    possible_values: Optional[List[int]] = None  # Assume present for ENUM type
    quantiles: Optional[
        List[float]
    ] = None  # Assume present for QUANTILE type and sorted
    min_value: Optional[float] = None
    max_value: Optional[float] = None


def no_op_feature():
    return NormalizationParameters(
        identify_types.CONTINUOUS, None, 0, 0, 1, None, None, None, None, None
    )


def identify_parameter(
    feature_name,
    values,
    max_unique_enum_values,
    quantile_size,
    quantile_k2_threshold,
    skip_box_cox,
    skip_quantiles,
    feature_type=None,
):
    if feature_type is None:
        feature_type = identify_types.identify_type(values, max_unique_enum_values)

    boxcox_lambda = None
    boxcox_shift = 0.0
    mean = 0.0
    stddev = 1.0
    mode = None
    possible_values = None
    quantiles = None
    assert feature_type in [
        identify_types.CONTINUOUS,
        identify_types.PROBABILITY,
        identify_types.BINARY,
        identify_types.ENUM,
        identify_types.CONTINUOUS_ACTION,
        identify_types.DO_NOT_PREPROCESS,
    ], "unknown type {}".format(feature_type)
    assert (
        len(values) >= MINIMUM_SAMPLES_TO_IDENTIFY
    ), "insufficient information to identify parameter"

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    mode = float(stats.mode(values).mode[0])

    if feature_type == identify_types.DO_NOT_PREPROCESS:
        mean = float(np.mean(values))
        values = values - mean
        stddev = max(float(np.std(values, ddof=1)), 1.0)
    if feature_type == identify_types.CONTINUOUS:
        if min_value == max_value:
            return no_op_feature()
        k2_original, p_original = stats.normaltest(values)

        # shift can be estimated but not in scipy
        boxcox_shift = float(min_value * -1)
        candidate_values, lambda_ = stats.boxcox(
            np.maximum(values + boxcox_shift, BOX_COX_MARGIN)
        )
        k2_boxcox, p_boxcox = stats.normaltest(candidate_values)
        logger.info(
            "Feature stats: original K2: {} P: {} Boxcox K2: {} P: {}".format(
                k2_original, p_original, k2_boxcox, p_boxcox
            )
        )
        if lambda_ < 0.9 or lambda_ > 1.1:
            # Lambda is far enough from 1.0 to be worth doing boxcox
            if k2_original > k2_boxcox * 10 and k2_boxcox <= quantile_k2_threshold:
                # The boxcox output is significantly more normally distributed
                # than the original data and is normal enough to apply
                # effectively.

                stddev = float(np.std(candidate_values, ddof=1))
                # Unclear whether this happens in practice or not
                if (
                    np.isfinite(stddev)
                    and stddev < BOX_COX_MAX_STDDEV
                    and not np.isclose(stddev, 0)
                ):
                    values = candidate_values
                    boxcox_lambda = float(lambda_)
        if boxcox_lambda is None or skip_box_cox:
            boxcox_shift = None
            boxcox_lambda = None
        if boxcox_lambda is not None:
            feature_type = identify_types.BOXCOX
        if (
            boxcox_lambda is None
            and k2_original > quantile_k2_threshold
            and (not skip_quantiles)
        ):
            feature_type = identify_types.QUANTILE
            quantiles = (
                np.unique(
                    mquantiles(
                        values,
                        np.arange(quantile_size + 1, dtype=np.float64)
                        / float(quantile_size),
                        alphap=0.0,
                        betap=1.0,
                    )
                )
                .astype(float)
                .tolist()
            )
            logger.info("Feature is non-normal, using quantiles: {}".format(quantiles))

    if (
        feature_type == identify_types.CONTINUOUS
        or feature_type == identify_types.BOXCOX
        or feature_type == identify_types.CONTINUOUS_ACTION
    ):
        mean = float(np.mean(values))
        values = values - mean
        stddev = max(float(np.std(values, ddof=1)), 1.0)
        if not np.isfinite(stddev):
            logger.info("Std. dev not finite for feature {}".format(feature_name))
            return None
        values /= stddev

    if feature_type == identify_types.ENUM:
        possible_values = np.unique(values.astype(int)).astype(int).tolist()

    return NormalizationParameters(
        feature_type,
        boxcox_lambda,
        boxcox_shift,
        mean,
        stddev,
        mode,
        possible_values,
        quantiles,
        min_value,
        max_value,
    )


def get_feature_start_indices(sorted_features, normalization_parameters):
    """ Returns the starting index for each feature in the output feature vector """
    start_indices = []
    cur_idx = 0
    for feature in sorted_features:
        np = normalization_parameters[feature]
        start_indices.append(cur_idx)
        if np.feature_type == identify_types.ENUM:
            cur_idx += len(np.possible_values)
        else:
            cur_idx += 1
    return start_indices


def sort_features_by_normalization(
    normalization_parameters: Dict[int, NormalizationParameters]
) -> Tuple[List[int], List[int]]:
    """
    Helper function to return a sorted list from a normalization map.
    Also returns the starting index for each feature type"""
    # Sort features by feature type
    sorted_features: List[int] = []
    feature_starts: List[int] = []
    assert isinstance(
        list(normalization_parameters.keys())[0], str
    ), "Normalization Parameters need to be str"
    for feature_type in identify_types.FEATURE_TYPES:
        feature_starts.append(len(sorted_features))
        for feature in sorted(normalization_parameters.keys()):
            norm = normalization_parameters[feature]
            if norm.feature_type == feature_type:
                sorted_features.append(feature)
    return sorted_features, feature_starts
