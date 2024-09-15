import torch
import torch.nn as nn
import math
from activation import softmax

class WeightedAttentionScores(nn.Module):
    def __init__(self, model_dimensionality, num_heads):
        super(self, WeightedAttentionScores).__init__()

        self.model_dimensionality = model_dimensionality
        self.num_heads = num_heads
        self.d_k = model_dimensionality // num_heads

    def dot_product_attention(self, query, key, mask):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probabilities = softmax.Softmax(x = attention_scores, dim=-1)

        return attention_probabilities
    
    def forward(self, value):
        output = torch.matmul(self.dot_product_attention, value)