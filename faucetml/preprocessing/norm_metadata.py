"""
Code taken from Facebook ReAgent and modified for Faucet ML.

Original copyright:
    Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

Original BSD License:
    https://github.com/facebookresearch/ReAgent/blob/master/LICENSE
"""

from typing import List, Dict

import numpy as np
import pandas as pd

from .normalization import identify_parameter, no_op_feature
from ..utils import get_logger

logger = get_logger(__name__)


def _get_single_feature_norm_metadata(
    feature_name: str,
    feature_value_list: List,
    skip_preprocessing: bool,
    feature_overrides: Dict[str, str],
    max_unique_enum_values: int,
    quantile_size: int,
    quantile_k2_threshold: int,
    skip_box_cox: int,
    skip_quantiles: int,
):
    logger.info("Got feature: {}".format(feature_name))
    feature_override = None

    if skip_preprocessing:
        feature_override = "DO_NOT_PREPROCESS"

    elif feature_overrides is not None:
        feature_override = feature_overrides.get(feature_name, None)

    feature_values = np.array(feature_value_list, dtype=np.float32)
    assert not (np.any(np.isinf(feature_values))), "Feature values contain infinity"
    assert not (
        np.any(np.isnan(feature_values))
    ), "Feature values contain nan (are there nulls in the feature values?)"
    normalization_parameters = identify_parameter(
        feature_name,
        feature_values,
        max_unique_enum_values,
        quantile_size,
        quantile_k2_threshold,
        skip_box_cox,
        skip_quantiles,
        feature_type=feature_override,
    )
    logger.info(
        "Feature {} normalization: {}\n".format(feature_name, normalization_parameters)
    )
    return normalization_parameters


def get_norm_metadata_dict(
    data_df: pd.DataFrame,
    exclude_features: List[str],
    skip_preprocessing: bool,
    feature_overrides: Dict[str, str],
    max_unique_enum_values: int,
    quantile_size: int,
    quantile_k2_threshold: int,
    skip_box_cox: int,
    skip_quantiles: int,
) -> Dict:
    exclude_features = set(exclude_features)
    output = {}
    for col, data in data_df.iteritems():
        if col in exclude_features:
            pass
        else:
            output[col] = _get_single_feature_norm_metadata(
                col,
                list(data),
                skip_preprocessing,
                feature_overrides,
                max_unique_enum_values,
                quantile_size,
                quantile_k2_threshold,
                skip_box_cox,
                skip_quantiles,
            )
    return output
