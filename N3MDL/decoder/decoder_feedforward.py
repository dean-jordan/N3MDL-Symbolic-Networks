import torch.nn as nn
from activation import relu
from layers import decoder_recurrent

class DecoderFeedForwardNetwork(nn.Module):
    def __init__(self):
        super(DecoderFeedForwardNetwork, self).__init__()