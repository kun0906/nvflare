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

import os.path
import pickle
from collections import Counter

import numpy as np
import torch
from pt_constants import PTConstants
from simple_network import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from data import CustomCIFAR10Dataset

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.abstract.model import (
    ModelLearnable,
    ModelLearnableKey,
    make_model_learnable,
    model_learnable_to_dxo,
)
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class PTLearner(Learner):
    def __init__(
        self,
        train_path='/users/kunyang/cifar10-hello-pt-10clients-2classes/data/client_8_airplane_train.pkl',
        test_path='/users/kunyang/cifar10-hello-pt-10clients-2classes/data/client_8_airplane_test.pkl',
        lr=0.01,
        epochs=5,
        exclude_vars=None,
        analytic_sender_id="analytic_sender",
    ):
        """Simple PyTorch Learner that trains and validates a simple network on the CIFAR10 dataset.

        Args:
            data_path (str): Path that the data will be stored at. Defaults to "~/data".
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            exclude_vars (list): List of variables to exclude during model loading.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
        """
        super().__init__()
        # self.writer = None
        # self.persistence_manager = None
        # self.default_train_conf = None
        # self.test_loader = None
        # self.test_data = None
        # self.n_iterations = None
        # self.train_loader = None
        # self.train_dataset = None
        # self.optimizer = None
        # self.loss = None
        # self.device = None
        # self.model = None

        self.train_path = train_path
        self.test_path = test_path
        self.lr = lr
        self.epochs = epochs
        self.exclude_vars = exclude_vars
        self.analytic_sender_id = analytic_sender_id

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # Training setup
        self.model = SimpleNetwork()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        # load train set
        self.logger.info(os.getcwd())
        with open(self.train_path, "rb") as f:
            # Get the size of the subset
            client_data = pickle.load(f)
        # Unpack client_data into separate lists
        client_images, _, client_targets = client_data
        client_images = np.array(client_images).astype('float32')
        client_targets = np.array(client_targets).astype('int')

        # Convert lists to tensors if needed
        # client_images = torch.tensor(client_images)
        # Convert to float32 if needed
        # client_images = client_images.astype('float32')

        # Convert to PyTorch tensor
        client_images = torch.tensor(client_images, dtype=torch.float32)
        client_targets = torch.tensor(client_targets)
        # Convert to PyTorch format: [batch_size, num_channels, height, width]
        client_images = client_images.permute(0, 3, 1, 2)

        # Instantiate the custom dataset
        self._train_subset = CustomCIFAR10Dataset(data=client_images, targets=client_targets, transform=None)

        self.train_loader = DataLoader(self._train_subset, batch_size=4, shuffle=True)
        self.n_iterations = len(self.train_loader)
        subset_size = len(self._train_subset)
        self.logger.info(f"Client {self.__class__.name}: Size of the training subset: {subset_size}")
        # Count the number of samples in each class within the subset
        class_counts = Counter()
        for label in self._train_subset.targets.numpy():
            class_counts[str(label)] += 1
        # Print the counts for each class
        for class_index, count in class_counts.items():
            self.logger.info(f"Class {class_index}: {count} samples")
        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self.default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self.default_train_conf
        )

        # load test set
        print(os.getcwd())
        with open(self.test_path, "rb") as f:
            # Get the size of the subset
            client_data = pickle.load(f)
        # Unpack client_data into separate lists
        client_images, _, client_targets = client_data
        client_images = np.array(client_images).astype('float32')
        client_targets = np.array(client_targets).astype('int')

        # Convert lists to tensors if needed
        # client_images = torch.tensor(client_images)
        # Convert to float32 if needed
        # client_images = client_images.astype('float32')

        # Convert to PyTorch tensor
        client_images = torch.tensor(client_images, dtype=torch.float32)
        client_targets = torch.tensor(client_targets)
        # Convert to PyTorch format: [batch_size, num_channels, height, width]
        client_images = client_images.permute(0, 3, 1, 2)

        # Instantiate the custom dataset
        self._test_subset = CustomCIFAR10Dataset(data=client_images, targets=client_targets, transform=None)

        self.test_loader = DataLoader(self._test_subset, batch_size=4, shuffle=True)
        # self._n_iterations = len(self._test_loader)
        subset_size = len(self._test_subset)
        print(f"Client {self.__class__.name}: Size of the testing subset: {subset_size}")
        # Count the number of samples in each class within the subset
        class_counts = Counter()
        for label in self._test_subset.targets.numpy():
            class_counts[str(label)] += 1
        # Print the counts for each class
        for class_index, count in class_counts.items():
            print(f"Class {class_index}: {count} samples")


        # Tensorboard streaming setup
        self.writer = parts.get(self.analytic_sender_id)  # user configuration from config_fed_client.json
        # if not self.writer:  # else use local TensorBoard writer only
        #     self.writer = SummaryWriter(fl_ctx.get_prop(FLContextKey.APP_ROOT))

    def get_weights(self):
        # Get the new state dict and send as weights
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        # self.logger.info(f'client 1: {weights}')
        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.n_iterations}
        )
        return outgoing_dxo.to_shareable()
    def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Get model weights
        try:
            dxo = from_shareable(data)
        except:
            self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Ensure data kind is weights.
        if not dxo.data_kind == DataKind.WEIGHTS:
            self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Convert weights to tensor. Run training
        torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
        # Set the model weights
        self.model.load_state_dict(state_dict=torch_weights)
        self.local_train(fl_ctx, abort_signal)

        # Check the abort_signal after training.
        # local_train returns early if abort_signal is triggered.
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Save the local model after training.
        self.save_local_model(fl_ctx)

        # Get the new state dict and send as weights
        new_weights = self.model.state_dict()
        new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=new_weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.n_iterations}
        )
        return outgoing_dxo.to_shareable()

    def local_train(self, fl_ctx, abort_signal):
        # Basic training
        current_round = fl_ctx.get_prop(FLContextKey.TASK_DATA).get_header("current_round")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                if abort_signal.triggered:
                    return

                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i % 3000 == 0:
                    self.log_info(
                        fl_ctx, f"Epoch: {epoch}/{self.epochs}, Iteration: {i}, " f"Loss: {running_loss/3000}"
                    )
                    running_loss = 0.0

                # # Stream training loss at each step
                # current_step = self.n_iterations * self.epochs * current_round + self.n_iterations * epoch + i
                # self.writer.add_scalar("train_loss", cost.item(), current_step)

            self.writer.add_scalar("train_loss", running_loss, epoch)
            # Stream validation accuracy at the end of each epoch
            metric = self.local_validate(abort_signal)
            self.writer.add_scalar("validation_accuracy", metric, epoch)

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self.default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self.exclude_vars)

        # Get the model parameters and create dxo from it
        dxo = model_learnable_to_dxo(ml)
        return dxo.to_shareable()

    def validate(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        model_owner = "?"
        try:
            try:
                dxo = from_shareable(data)
            except:
                self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            # Ensure data_kind is weights.
            if not dxo.data_kind == DataKind.WEIGHTS:
                self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            if isinstance(dxo.data, ModelLearnable):
                dxo.data = dxo.data[ModelLearnableKey.WEIGHTS]

            # Extract weights and ensure they are tensor.
            model_owner = data.get_header(AppConstants.MODEL_OWNER, "?")
            weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

            self.model.load_state_dict(weights)

            # Get validation accuracy
            val_accuracy = self.local_validate(abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            self.log_info(
                fl_ctx,
                f"Accuracy when validating {model_owner}'s model on"
                f" {fl_ctx.get_identity_name()}"
                f"s data: {val_accuracy}",
            )

            dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
            return dxo.to_shareable()
        except:
            self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def local_validate(self, abort_signal):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)

                _, pred_label = torch.max(output, 1)

                correct += (pred_label == labels).sum().item()
                total += images.size()[0]
            metric = correct / float(total)
        return metric

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)