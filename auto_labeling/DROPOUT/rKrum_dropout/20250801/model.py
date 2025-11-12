
import collections
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, Subset


class FNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # From 1 channel to 16 channels
        # self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        #
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        # self.conv21 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        #
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # self.conv31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #
        # self.fc1 = nn.Linear(64 * 3 * 3, 512)  # Adjust the dimensions after the convolution layers
        # self.fc2 = nn.Linear(512, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # # self.sigmoid = nn.Sigmoid()
        #
        # self.transform = nn.Sequential(
        #     nn.Linear(28 * 28 + num_classes, 784),
        #     nn.LeakyReLU(0.2),
        # )

        self.fc11 = nn.Linear(100, 32)
        self.fc21 = nn.Linear(32, 16)
        self.fc22 = nn.Linear(16, 8)
        self.fc33 = nn.Linear(8, num_classes)

    def forward(self, x):
        model_type = 'mlp'

        x = self.leaky_relu(self.fc11(x))
        x = self.leaky_relu(self.fc21(x))
        x = self.leaky_relu(self.fc22(x))
        x = self.fc33(x)

        return x
