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

################# Testing for transform_tensor_to_finite and transform_finite_to_tensor ########################
from fedml_api.distributed.lightsecagg.utils import transform_tensor_to_finite, transform_finite_to_tensor

# the test is to check if the conversion between two fields works 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
model = NeuralNetwork()

model = model.state_dict()
print(model)
p = 2**31-1
model = transform_tensor_to_finite(model, p, 15)
print(model)

model = transform_finite_to_tensor(model, p, 15)
print(model)