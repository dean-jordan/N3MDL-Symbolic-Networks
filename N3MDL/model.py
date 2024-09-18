import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

class N3MDLFullNetwork(nn.Module):
    def __init__(self, model_dimensionality, source_vocab_size, target_vocab_size, num_heads,
                                               num_layers_encoder, num_layers_decoder, num_central_subnetworks,
                                               num_layers_central_subnetworks, central_subnetwork_dimensionality,
                                               ff_dimensionality_encoder, ff_dimensionality_decoder, max_sequence_length, dropout):
        super(N3MDLFullNetwork, self).__init__()