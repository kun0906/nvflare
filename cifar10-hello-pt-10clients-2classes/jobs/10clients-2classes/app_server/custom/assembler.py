# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext

from nvflare.app_common.aggregators.assembler import Assembler


class MY_Assembler(Assembler):
    """Assembler class for aggregation functionality
    This defines the functionality of assembling the collected submissions
    for CollectAndAssembleAggragator
    """

    def __init__(self, data_kind: str):
        super().__init__(data_kind)
        self.expected_data_kind = data_kind
        self.logger.debug(f"expected data kind: {self.expected_data_kind}")
        self._collection: dict = {}

    def initialize(self, fl_ctx: FLContext):
        pass

    @property
    def collection(self):
        return self._collection

    def get_expected_data_kind(self):
        return self.expected_data_kind

    def get_model_params(self, dxo: DXO) -> dict:
        """Connects the assembler's _collection with CollectAndAssembleAggregator
        Get the collected parameters from the main aggregator
        Return:
            A dict of parameters needed for further assembling
        """
        # raise NotImplementedError
        return dxo

    def assemble(self, data: Dict[str, dict], fl_ctx: FLContext) -> DXO:
        """Assemble the collected submissions.
        This will be specified according to the specific algorithm
        E.g. global svm round on the collected local supporting vectors;
        global k-means step on the local centroids and counts

        Return:
            A DXO containing all information ready to be returned to clients
        """
        # raise NotImplementedError
        agg = {}
        # key = data.keys()[0]
        first_key = next(iter(data))
        data_kind = data[first_key].data_kind
        meta = data[first_key].meta
        for para_key, para_val in data[first_key].data.items():
            para_vals = None
            for site_, dxo_ in data.items():
                para_val2 = dxo_.data[para_key]
                # if 'bias' in para_key:  # or len(para_val2.shape) == 1:
                #     para_val2 = para_val2.reshape((1, -1))
                if para_vals is None:  # para_vals is None
                    para_vals = para_val2[np.newaxis, ...]
                else:
                    # para_vals = np.concatenate([para_vals, para_val2], axis=0)
                    # Stack A and B along a new axis (axis=0)
                    self.logger.debug(f"para_key:{para_key}, para_vals.shape:{para_vals.shape}, para_val2.shape:{para_val2.shape}")
                    para_vals = np.concatenate([para_vals, para_val2[np.newaxis, ...]], axis=0)
                    # para_vals = np.vstack([para_vals, para_val2])
            # compute the coordinate median
            para_vals = np.median(para_vals, axis=0)
            # if 'bias' in para_key:
            #     para_vals = para_vals.reshape((-1,))
            agg[para_key] = para_vals

        return DXO(data_kind=data_kind, data=agg, meta=meta)

    def reset(self) -> None:
        # Reset parameters for next round,
        # This will be performed at the end of each aggregation round,
        # it can include, but not limited to, clearing the _collection
        self._collection = {}
