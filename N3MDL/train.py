import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from network import network

# Dimensionality of Model Embeddings
model_dimensionality = 512

# Source and Target Vocab Embedding Sizes
source_vocab_size = 100
target_vocab_size = 100

# Number of Attention Heads in Multi-Head Attention
num_heads = 8

# Number of Layers in Both Encoder and Decoder
num_layers_encoder = 6
num_layers_decoder = 6

# Number of Subnetworks in Central Networks
num_central_subnetworks = 4

# Number of Layers in each Central Subnetwork
num_layers_central_subnetworks = 6

# Dimensionality of each Central Subnetwork
central_subnetwork_dimensionality = 512

# Dimensionality of Feed-Forward Network Within Encoder/Decoder
ff_dimensionality_encoder = 512
ff_dimensionality_decoder = 512

# Maximum Sequence Length in Positional Encoding
max_sequence_length = 100

# Dropout Regularization Rate
dropout = 0.1

transformer_network = network.N3MDLFullNetwork(model_dimensionality, source_vocab_size, target_vocab_size, num_heads,
                                               num_layers_encoder, num_layers_decoder, num_central_subnetworks,
                                               num_layers_central_subnetworks, central_subnetwork_dimensionality,
                                               ff_dimensionality_encoder, ff_dimensionality_decoder, max_sequence_length, dropout)