import torch.nn as nn

class CombineAttentionHeads(nn.Module):
    def __init__(self, model_dimensionality, num_heads):
        super(self, CombineAttentionHeads).__init__()

        self.model_dimensionality = model_dimensionality
        self.num_heads = num_heads
        self.d_k = model_dimensionality // num_heads

    def forward(self, x):
        batch_size, sequence_length, d_k = x.size()
        return x.view(batch_size, sequence_length, d_k, self.num_heads).transpose(1, 2)