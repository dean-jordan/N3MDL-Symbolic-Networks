import torch.nn as nn
from activation import relu
from layers import encoder_recurrent

class EncoderFeedForwardNetwork(nn.Module):
    def __init__(self):
        super(EncoderFeedForwardNetwork, self).__init__()