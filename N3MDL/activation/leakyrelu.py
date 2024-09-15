import torch
import torch.nn as nn

class LeakyReLU(nn.Module):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def foward(self, x, negative_slope):
        if x > 0:
            return x
        else:
            return x * negative_slope