import torch.nn as nn
from attention import attention

class DecoderAttentionModule(nn.Module):
    def __init__(self, model_dimensionality: int, num_heads: int, dropout):
        super(self, DecoderAttentionModule).__init__()

        self.attention = attention.AttentionMechanism(model_dimensionality, num_heads)
        self.normalization = nn.LayerNorm(model_dimensionality)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.attention(x, x, x, mask)
        x = self.normalization(x + self.dropout(attention_output))