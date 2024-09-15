import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

class EncoderAttention(nn.Module):
    def __init__(self):
        super(self, EncoderAttention).__init__()
