import argparse
import logging
import os
import random
import socket
import sys
import traceback

import numpy as np
import psutil
import setproctitle
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

################# Testing for model_dimension ########################
from fedml_api.distributed.lightsecagg.utils import transform_tensor_to_finite

# transform_tensor_to_finite is to covert a model in from real (in tensor) to finite (in numpy) 
# 1
a = np.ones(5)
b = torch.from_numpy(a)
out = transform_tensor_to_finite({"1":b}, 1, 1)
print(out)

import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

