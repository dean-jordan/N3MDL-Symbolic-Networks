import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

class SubnetworkIdentityBlock(nn.Module):
    def __init__(self):
        super(SubnetworkIdentityBlock, self).__init__()

        