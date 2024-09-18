import torch.nn as nn
from activation import relu
from layers import decoder_recurrent

class DecoderFeedForwardNetwork(nn.Module):
    def __init__(self):
        super(DecoderFeedForwardNetwork, self).__init__()

        self.recurrent1 = decoder_recurrent.DecoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.recurrent2 = decoder_recurrent.DecoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.recurrent3 = decoder_recurrent.DecoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.recurrent4 = decoder_recurrent.DecoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.recurrent5 = decoder_recurrent.DecoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.activation = relu.ReLU()

    def forward(self, x):
        self.recurrent5(
            self.activation(
                self.recurrent4(
                    self.activation(
                        self.recurrent3(
                            self.activation(
                                self.recurrent2(
                                    self.activation(
                                        self.recurrent1(
                                            self.activation(x)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )