# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Union, Dict

import numpy as np
from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.response_processor import ResponseProcessor


def aggregator(data: Dict[str, dict], fl_ctx: FLContext, method='median') -> DXO:
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
                # self.logger.debug(
                #     f"para_key:{para_key}, para_vals.shape:{para_vals.shape}, para_val2.shape:{para_val2.shape}")
                para_vals = np.concatenate([para_vals, para_val2[np.newaxis, ...]], axis=0)
                # para_vals = np.vstack([para_vals, para_val2])
        if method == 'median':
            # compute the coordinate median
            para_vals = np.median(para_vals, axis=0)
        else:
            para_vals = np.mean(para_vals, axis=0)
        agg[para_key] = para_vals

    return DXO(data_kind=data_kind, data=agg, meta=meta)


class BroadcastAndProcess(Controller):
    def __init__(
            self,
            processor: Union[str, ResponseProcessor],
            task_name: str,
            min_responses_required: int = 0,
            wait_time_after_min_received: int = 10,
            timeout: int = 0,
            clients=None,
    ):
        """This controller broadcast a task to specified clients to collect responses, and uses the
        ResponseProcessor object to process the client responses.

        Args:
            processor: the processor that implements logic for client responses and final check.
            It must be a component id (str), or a ResponseProcessor object.
            task_name: name of the task to be sent to client to collect responses
            min_responses_required: min number responses required from clients. 0 means all.
            wait_time_after_min_received: how long to wait after min responses are received from clients
            timeout: timeout of the task. 0 means never time out
            clients: list of clients to send the task to. None means all clients.
        """
        Controller.__init__(self)
        if not (isinstance(processor, str) or isinstance(processor, ResponseProcessor)):
            raise TypeError(f"value of processor must be a str or ResponseProcessor but got {type(processor)}")

        self.processor = processor
        self.task_name = task_name
        self.min_responses_required = min_responses_required
        self.wait_time_after_min_received = wait_time_after_min_received
        self.timeout = timeout
        self.clients = clients

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing BroadcastAndProcess.")
        if isinstance(self.processor, str):
            checker_id = self.processor

            # the processor is a component id - get the processor component
            engine = fl_ctx.get_engine()
            if not engine:
                self.system_panic("Engine not found. BroadcastAndProcess exiting.", fl_ctx)
                return

            self.processor = engine.get_component(checker_id)
            if not isinstance(self.processor, ResponseProcessor):
                self.system_panic(
                    f"component {checker_id} must be a ResponseProcessor type object but got {type(self.processor)}",
                    fl_ctx,
                )

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        task_data = self.processor.create_task_data(self.task_name, fl_ctx)
        if not isinstance(task_data, Shareable):
            self.system_panic(
                f"ResponseProcessor {type(self.processor)} failed to return valid task data: "
                f"expect Shareable but got {type(task_data)}",
                fl_ctx,
            )
            return

        task = Task(
            name=self.task_name,
            data=task_data,
            timeout=self.timeout,
            result_received_cb=self._process_client_response,
        )

        self.broadcast_and_wait(
            task=task,
            wait_time_after_min_received=self.wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
            targets=self.clients,
            min_responses=self.min_responses_required,
        )

        # Add your own initialize method
        aggr_result = aggregator(self.processor.collections, fl_ctx, method=self.processor.weight_method)
        self.processor.final_weights = aggr_result.data

        success = self.processor.final_process(fl_ctx)
        if not success:
            self.system_panic(reason=f"ResponseProcessor {type(self.processor)} failed final check!", fl_ctx=fl_ctx)

    def _process_client_response(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        task = client_task.task
        response = client_task.result
        client = client_task.client

        ok = self.processor.process_client_response(
            client=client, task_name=task.name, response=response, fl_ctx=fl_ctx
        )

        # Cleanup task result
        client_task.result = None

        if not ok:
            self.system_panic(
                reason=f"ResponseProcessor {type(self.processor)} failed to check client {client.name}", fl_ctx=fl_ctx
            )

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(
            self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass


# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.client import Client
from nvflare.apis.dxo import DataKind, from_shareable, DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.response_processor import ResponseProcessor
from nvflare.app_common.app_constant import AppConstants


class WeightMethod(object):
    FIRST = "first"
    CLIENT = "client"


class GlobalWeightsInitializer(ResponseProcessor):
    def __init__(
            self,
            weights_prop_name: str = AppConstants.GLOBAL_MODEL,
            weight_method: str = WeightMethod.FIRST,
            client_name: str = None,
    ):
        """Set global model weights based on specified weight setting method.

        Args:
            weights_prop_name: name of the prop to be set into fl_ctx for the determined global weights
            weight_method: the method to select final weights: one of "first", "client"
            client_name: the name of the client to be used as the weight provider

        If weight_method is "first", then use the weights reported from the first client;
        If weight_method is "client", then only use the weights reported from the specified client.
        """
        # if weight_method not in [WeightMethod.FIRST, WeightMethod.CLIENT]:
        #     raise ValueError(f"invalid weight_method '{weight_method}'")
        # if weight_method == WeightMethod.CLIENT and not client_name:
        #     raise ValueError(f"client name not provided for weight method '{WeightMethod.CLIENT}'")
        # if weight_method == WeightMethod.CLIENT and not isinstance(client_name, str):
        #     raise ValueError(
        #         f"client name should be a single string for weight method '{WeightMethod.CLIENT}' but it is {client_name} "
        #     )

        ResponseProcessor.__init__(self)
        self.weights_prop_name = weights_prop_name
        self.weight_method = weight_method
        self.client_name = client_name
        self.final_weights = None
        self.collections = {}

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        """Create the data for the task to be sent to clients to collect their weights

        Args:
            task_name: name of the task
            fl_ctx: the FL context

        Returns: task data

        """
        # reset internal state in case this processor is used multiple times
        self.final_weights = None
        return Shareable()

    def process_client_response(self, client: Client, task_name: str, response: Shareable, fl_ctx: FLContext) -> bool:
        """Process the weights submitted by a client.

        Args:
            client: the client that submitted the response
            task_name: name of the task
            response: submitted data from the client
            fl_ctx: FLContext

        Returns:
            boolean to indicate if the client data is acceptable.
            If not acceptable, the control flow will exit.

        """
        if not isinstance(response, Shareable):
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: " f"response must be Shareable but got {type(response)}",
            )
            return False

        try:
            dxo = from_shareable(response)
        except Exception:
            self.log_exception(fl_ctx, f"bad response from client {client.name}: " f"it does not contain DXO")
            return False

        if dxo.data_kind != DataKind.WEIGHTS:
            self.log_error(
                fl_ctx,
                f"bad response from client {client.name}: "
                f"data_kind should be DataKind.WEIGHTS but got {dxo.data_kind}",
            )
            return False

        weights = dxo.data
        if not weights:
            self.log_error(fl_ctx, f"No model weights found from client {client.name}")
            return False

        if not self.final_weights and (
                self.weight_method == WeightMethod.FIRST
                or (self.weight_method == WeightMethod.CLIENT and client.name == self.client_name)
        ):
            self.final_weights = weights
        self.collections[client.name] = dxo
        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        """Perform the final check on all the received weights from the clients.

        Args:
            fl_ctx: FLContext

        Returns:
            boolean indicating whether the final response processing is successful.
            If not successful, the control flow will exit.
        """
        if not self.final_weights:
            self.log_error(fl_ctx, "no weights available from clients")
            return False

        # must set sticky to True so other controllers can get it!
        fl_ctx.set_prop(self.weights_prop_name, make_model_learnable(self.final_weights, {}), private=True, sticky=True)
        return True


# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Union

from nvflare.app_common.app_constant import AppConstants


class MY_InitializeGlobalWeights(BroadcastAndProcess):
    def __init__(
            self,
            task_name: str = AppConstants.TASK_GET_WEIGHTS,
            min_responses_required: int = 0,
            wait_time_after_min_received: int = 0,
            task_timeout: int = 0,
            weights_prop_name=AppConstants.GLOBAL_MODEL,
            weight_method: str = "median",  # WeightMethod.FIRST,
            weights_client_name: Union[str, List[str], None] = None,
    ):
        """A controller for initializing global model weights based on reported weights from clients.

        Args:
            task_name: name of the task to be sent to clients to collect their model weights
            min_responses_required: min number of responses required. 0 means all clients.
            wait_time_after_min_received: how long (secs) to wait after min responses are received
            task_timeout: max amount of time to wait for the task to end. 0 means never time out.
            weights_prop_name: name of the FL Context property to store the global weights
            weight_method: method for determining global model weights. Defaults to `WeightMethod.FIRST`.
            weights_client_name: name of the client if the method is "client". Defaults to None.
                If `None`, the task will be sent to all clients (to be used with `weight_method=WeightMethod.FIRST`).
                If list of client names, the task will be only be sent to the listed clients.
        """

        if isinstance(weights_client_name, str):
            clients = [weights_client_name]
        elif isinstance(weights_client_name, list):
            clients = weights_client_name
        else:
            clients = None

        BroadcastAndProcess.__init__(
            self,
            processor=GlobalWeightsInitializer(
                weights_prop_name=weights_prop_name, weight_method=weight_method, client_name=weights_client_name
            ),
            task_name=task_name,
            min_responses_required=min_responses_required,
            wait_time_after_min_received=wait_time_after_min_received,
            timeout=task_timeout,
            clients=clients,
        )
