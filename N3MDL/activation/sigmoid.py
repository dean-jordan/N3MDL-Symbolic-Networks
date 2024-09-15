import torch
import torch.nn as nn

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))