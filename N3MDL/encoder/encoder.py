import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from encoder_block import EncoderBlock
from positional_encoding import EncoderPositionalEncodingModule

class EncoderEnsemble(nn.Module):
    def __init__(self, num_layers, num_encoders):
        super(EncoderEnsemble, self).__init__()

        self.encoder_block = EncoderBlock(model_dimensionality=8192, num_heads=16, feedforward_dimensionality=8192, dropout=0.1,
                                     hidden_dimensionality=8192)
        
    def create_encoder_ensemble(self, num_layers, num_encoders, model_dimensionality, dropout, source_vocab_size):
        num_layers = 8
        num_encoders = 2

        self.encoder_embedding = nn.Embedding(source_vocab_size, model_dimensionality)

        for _ in range(num_layers):
            self.encoder_layers = nn.ModuleList([self.encoder_block])

        for _ in range(num_encoders):
            self.encoder_number = nn.ModuleList([self.encoder_layers])

        self.fully_connected = nn.Linear(model_dimensionality)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (source != 0).unsqueeze(1).unsqueeze(3)

        sequence_length = target.size(1)

        nopeak_mask = (1 - torch.triu(torch.ones(1, sequence_length, sequence_length), diagonal=1)).bool()

        target_mask = target_mask & nopeak_mask

        return source_mask, target_mask
    
    def forward(self, source):
        source_mask = self.generate_mask(source)
        source_embedded = self.dropout(EncoderPositionalEncodingModule(source))

        encoder_output = source_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, source_mask)

        output = self.fully_connected(encoder_output)
        return output