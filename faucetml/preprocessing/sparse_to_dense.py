"""
Code taken from Facebook ReAgent and modified for Gradient.

Original copyright:
    Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

Original BSD License:
    https://github.com/facebookresearch/ReAgent/blob/master/LICENSE
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd
import torch

from . import normalization


logger = logging.getLogger(__name__)


class SparseToDenseProcessor:
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ):
        self.sorted_features = sorted_features
        self.set_missing_value_to_zero = set_missing_value_to_zero

    def __call__(self, sparse_data):
        return self.process(sparse_data)


class PandasSparseToDenseProcessor(SparseToDenseProcessor):
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ):
        super().__init__(sorted_features, set_missing_value_to_zero)
        self.feature_to_index: Dict[int, int] = {}
        for i, f in enumerate(sorted_features):
            self.feature_to_index[f] = i

    def process(self, sparse_data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        missing_value = normalization.MISSING_VALUE
        if self.set_missing_value_to_zero:
            missing_value = 0.0
        state_features_df = sparse_data.fillna(missing_value)

        # Add columns identified by normalization, but not present in batch
        for col in self.sorted_features:
            if col not in state_features_df.columns:
                state_features_df[col] = missing_value

        values = torch.from_numpy(
            state_features_df[self.sorted_features].values
        ).float()
        if self.set_missing_value_to_zero:
            # When we set missing values to 0, we don't know what is and isn't missing
            presence = values != 0.0
        else:
            presence = values != missing_value
        return values, presence
