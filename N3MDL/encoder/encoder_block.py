import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from encoder_attention import EncoderAttentionModule
from encoder_feedforward import EncoderFeedForwardNetwork

class EncoderBlock(nn.Module):
    def __init__(self, model_dimensionality, num_heads, feedforward_dimensionality, dropout, hidden_dimensionality):
        super(EncoderBlock, self).__init__()

        self.attention_module = EncoderAttentionModule(model_dimensionality, num_heads, dropout)
        self.feedforward_network = EncoderFeedForwardNetwork(model_dimensionality, hidden_dimensionality, feedforward_dimensionality)
        self.layer_normalization1 = nn.LayerNorm(model_dimensionality)
        self.layer_normalization2 = nn.LayerNorm(model_dimensionality)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.attention_module(x, x, x, mask)
        x = self.layer_normalization1(x + self.dropout(attention_output))
        feedforward_output = self.feedforward_network(x)
        x = self.layer_normalization2(x + self.dropout(feedforward_output))
        return x