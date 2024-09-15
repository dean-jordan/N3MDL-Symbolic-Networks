import torch.nn as nn

class SplitAttentionHeads(nn.Module):
    def __init__(self, model_dimensionality, num_heads):
        super(self, SplitAttentionHeads).__init__()

        self.model_dimensionality = model_dimensionality
        self.num_heads = num_heads
        self.d_k = model_dimensionality // num_heads

    def forward(self, x):
        batch_size, _, sequence_length = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.model_dimensionality)