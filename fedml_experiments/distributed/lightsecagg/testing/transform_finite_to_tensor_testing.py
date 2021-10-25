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
from fedml_api.distributed.lightsecagg.utils import transform_finite_to_tensor
from fedml_api.distributed.lightsecagg.utils import transform_tensor_to_finite


# transform_finite_to_tensor is to convert a model from finite (in numpy) to real (in tensor)   

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

p = 3
q = 11
def check(d, p, q):
    source = d
    inter = transform_tensor_to_finite(d, p, q)
    target = transform_finite_to_tensor(inter, p, q)
    for key in inter:
        print(type(inter[key]))
        assert isinstance(inter[key], np.ndarray)
    for key in source:
        assert key in target
        assert torch.equal(source[key], target[key])

check(net.state_dict(), p, q)
check(optimizer.state_dict(), p, q)

print("tensor/finite passed")
