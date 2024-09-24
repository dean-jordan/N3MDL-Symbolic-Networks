import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from decoder_block import DecoderBlock

class DecoderEnsemble(nn.Module):
    def __init__(self, num_layers, ):
        super(DecoderEnsemble, self).__init__()



