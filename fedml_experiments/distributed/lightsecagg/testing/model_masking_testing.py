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

################# Testing for model_masking ########################
from fedml_api.distributed.lightsecagg.mpc_function import model_masking

# model_masking is to mask model by adding the local mask
# goal: add the local mask to the local model(weight)
# input: weights_finite(dict: each value is finite in numpy)
#        dimensions(list)
#        local_mask(int numpy array)
#        output: weights_finite(dict: each value is finite in numpy)

weights = {"1": np.arange(9).reshape((3,3))}
dim = [3,3]
local = np.zeros(9).reshape((3,3))
output = model_masking(weights, dim, local)
assert isinstance(output, dict)
assert output["1"] == np.arange(9).reshape((3,3))
