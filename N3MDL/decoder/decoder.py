import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from decoder_block import DecoderBlock
from positional_encoder import DecoderPositionalEncodingModule

class DecoderEnsemble(nn.Module):
    def __init__(self):
        super(DecoderEnsemble, self).__init__()

        self.decoder_block = DecoderBlock(model_dimensionality=8192, num_heads=16, feedforward_dimensionality=8192,
                                      dropout=0.1, hidden_dimensionality=8192)
        
    def create_decoder_ensemble(self, num_decoders, num_layers, model_dimensionality, dropout, target_vocab_size):
        num_decoders = 2
        num_layers = 8
        model_dimensionality = 8192
        dropout = 0.1

        self.decoder_embedding = nn.Embedding(target_vocab_size, model_dimensionality)

        for _ in range(num_layers):
            self.decoder_layers = nn.ModuleList([self.decoder_block])
            
        for _ in range(num_decoders):
            self.decoder_number = nn.ModuleList([self.decoder_layers()])

        self.fully_connected = nn.Linear(model_dimensionality)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (source != 0).unsqueeze(1).unsqueeze(3)

        sequence_length = target.size(1)

        nopeak_mask = (1 - torch.triu(torch.ones(1, sequence_length, sequence_length), diagonal=1)).bool()

        target_mask = target_mask & nopeak_mask

        return source_mask, target_mask
    
    def forward(self, target, encoder_output, source_mask):
        target_mask = self.generate_mask(target)
        target_embedded = self.dropout(DecoderPositionalEncodingModule(self.decoder_embedding(target)))

        decoder_output = target_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, source_mask, target_mask)

        output = self.fully_connected(decoder_output)
        return output
