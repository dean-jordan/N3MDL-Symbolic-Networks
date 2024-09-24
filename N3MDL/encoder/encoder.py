import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from encoder_block import EncoderBlock

class EncoderEnsemble(nn.Module):
    def __init__(self, num_layers, num_encoders):
        super(EncoderEnsemble, self).__init__()

        encoder_block = EncoderBlock(model_dimensionality=8192, num_heads=16, feedforward_dimensionality=8192, dropout=0.1,
                                     hidden_dimensionality=8192)