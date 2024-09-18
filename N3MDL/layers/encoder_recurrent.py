import torch
import torch.nn as nn

input_size = 2
hidden_size = 3
output_size = 2

class EncoderRecurrentLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(EncoderRecurrentLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(x + hidden_state)

        output = self.h2o(hidden_size)

        return output, hidden_state
    
    def init_zero_hidden(self, batch_size = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)