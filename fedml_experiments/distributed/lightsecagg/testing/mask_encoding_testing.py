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

################# Testing for mask_encoding ########################
from fedml_api.distributed.lightsecagg.mpc_function import LLC_encoding_with_points
from fedml_api.distributed.lightsecagg.mpc_function import LLC_decoding_with_points

# mask encoding is to mask the local mask via LCC encoding
# goal: encode the local generated mask via LCC encoding
# input: total_dimension (int)
#        num_clients (int)
#        targeted_number_active_clients (int)
#        privacy_guarantee (int)
#        prime_number (int)
#       local_mask (int numpy array)
# output: encoded_mask_set (int numpy array)

local = np.zeros(9, dtype=int).reshape((3,3))
print(local)
output = mask_encoding(3, 3, 2, 1, 2, local)
assert isinstance(output, np.array)
print(output)
