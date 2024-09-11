# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.security.logging import secure_format_exception


class LearnerExecutor(Executor):
    def __init__(
        self,
        learner_id,
        train_task=AppConstants.TASK_TRAIN,
        submit_model_task=AppConstants.TASK_SUBMIT_MODEL,
        validate_task=AppConstants.TASK_VALIDATION,
        pre_train_task_name=AppConstants.TASK_GET_WEIGHTS,
    ):
        """Key component to run learner on clients.

        Args:
            learner_id (str): id of the learner object
            train_task (str, optional): task name for train. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): task name for submit model. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): task name for validation. Defaults to AppConstants.TASK_VALIDATION.
        """
        super().__init__()
        self.learner_id = learner_id
        self.learner = None
        self.train_task = train_task
        self.submit_model_task = submit_model_task
        self.validate_task = validate_task
        self.pre_train_task_name = pre_train_task_name
        self.is_initialized = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABORT_TASK:
            try:
                if self.learner:
                    if not self.unsafe:
                        self.learner.abort(fl_ctx)
                    else:
                        self.log_warning(fl_ctx, f"skipped abort of unsafe learner {self.learner.__class__.__name__}")
            except Exception as e:
                self.log_exception(fl_ctx, f"learner abort exception: {secure_format_exception(e)}")
        elif event_type == EventType.END_RUN:
            if not self.unsafe:
                self.finalize(fl_ctx)
            elif self.learner:
                self.log_warning(fl_ctx, f"skipped finalize of unsafe learner {self.learner.__class__.__name__}")

    def initialize(self, fl_ctx: FLContext):
        try:
            engine = fl_ctx.get_engine()
            self.learner = engine.get_component(self.learner_id)
            if not isinstance(self.learner, Learner):
                raise TypeError(f"learner must be Learner type. Got: {type(self.learner)}")
            self.learner.initialize(engine.get_all_components(), fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"learner initialize exception: {secure_format_exception(e)}")
            raise e

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")
        if not self.is_initialized:
            self.is_initialized = True
            self.initialize(fl_ctx)

        if task_name == self.pre_train_task_name:
            # Get the new state dict and send as weights
            return self.learner.get_weights()
        elif task_name == self.train_task:
            return self.train(shareable, fl_ctx, abort_signal)
        elif task_name == self.submit_model_task:
            return self.submit_model(shareable, fl_ctx)
        elif task_name == self.validate_task:
            return self.validate(shareable, fl_ctx, abort_signal)
        else:
            self.log_error(fl_ctx, f"Could not handle task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_debug(fl_ctx, f"train abort signal: {abort_signal.triggered}")

        shareable.set_header(AppConstants.VALIDATE_TYPE, ValidateType.BEFORE_TRAIN_VALIDATE)
        validate_result: Shareable = self.learner.validate(shareable, fl_ctx, abort_signal)

        train_result = self.learner.train(shareable, fl_ctx, abort_signal)
        if not (train_result and isinstance(train_result, Shareable)):
            return make_reply(ReturnCode.EMPTY_RESULT)

        # if the learner returned the valid BEFORE_TRAIN_VALIDATE result, set the INITIAL_METRICS in
        # the train result, which can be used for best model selection.
        if (
            validate_result
            and isinstance(validate_result, Shareable)
            and validate_result.get_return_code() == ReturnCode.OK
        ):
            try:
                metrics_dxo = from_shareable(validate_result)
                train_dxo = from_shareable(train_result)
                train_dxo.meta[MetaKey.INITIAL_METRICS] = metrics_dxo.data.get(MetaKey.INITIAL_METRICS, 0)
                return train_dxo.to_shareable()
            except ValueError:
                return train_result
        else:
            return train_result

    def submit_model(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        model_name = shareable.get_header(AppConstants.SUBMIT_MODEL_NAME)
        submit_model_result = self.learner.get_model_for_validation(model_name, fl_ctx)
        if submit_model_result and isinstance(submit_model_result, Shareable):
            return submit_model_result
        else:
            return make_reply(ReturnCode.EMPTY_RESULT)

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_debug(fl_ctx, f"validate abort_signal {abort_signal.triggered}")

        shareable.set_header(AppConstants.VALIDATE_TYPE, ValidateType.MODEL_VALIDATE)
        validate_result: Shareable = self.learner.validate(shareable, fl_ctx, abort_signal)
        if validate_result and isinstance(validate_result, Shareable):
            return validate_result
        else:
            return make_reply(ReturnCode.EMPTY_RESULT)

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.learner:
                self.learner.finalize(fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"learner finalize exception: {secure_format_exception(e)}")
