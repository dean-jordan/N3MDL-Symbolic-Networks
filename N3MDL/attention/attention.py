import torch.nn as nn
from combine_heads import CombineAttentionHeads
from split_heads import SplitAttentionHeads
from weight import WeightedAttentionScores

class AttentionMechanism(nn.Module):
    def __init__(self, model_dimensionality, num_heads):
        super(self, AttentionMechanism).__init__()

        self.model_dimensionality = model_dimensionality
        self.num_heads = num_heads
        self.d_k = model_dimensionality // num_heads

        self.query = nn.Linear(model_dimensionality, model_dimensionality)
        self.key = nn.Linear(model_dimensionality, model_dimensionality)
        self.value = nn.Linear(model_dimensionality, model_dimensionality)
        self.output = nn.Linear(model_dimensionality, model_dimensionality)

    def forward(self, query, key, value, mask=None):
        query = SplitAttentionHeads(self.query(query))
        key = SplitAttentionHeads(self.key(key))
        value = SplitAttentionHeads(self.value(value))

        attention_output = WeightedAttentionScores(query, key, value, mask)

        output = self.output(CombineAttentionHeads(attention_output))