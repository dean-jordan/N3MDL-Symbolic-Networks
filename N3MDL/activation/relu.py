import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU).__init__()

    def forward(self):
        x = torch.maximum(x, 0.0)
        return x