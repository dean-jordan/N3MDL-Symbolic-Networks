import torch
import torch.nn as nn
import math

class AttentionPositionalEncodingModule(nn.Module):
    def __init__(self, model_dimensionality, max_sequence_length):
        super(AttentionPositionalEncodingModule, self).__init__()

        encoding_tensors = torch.zeros(max_sequence_length, model_dimensionality)
        positional_tensors = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        divided_term = torch.exp(torch.arange(0, model_dimensionality, 2).float() * -(math.log(10000.0) / model_dimensionality))

        encoding_tensors[:, 0::2] = torch.sin(positional_tensors * divided_term)
        encoding_tensors[:, 1::2] = torch.cos(positional_tensors * divided_term)

        self.register_buffer('encoding_tensors', encoding_tensors.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding_tensors[:, :x.size(1)]