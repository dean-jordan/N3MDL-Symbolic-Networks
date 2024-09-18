import torch.nn as nn
from activation import relu
from layers import encoder_recurrent

class EncoderFeedForwardNetwork(nn.Module):
    def __init__(self):
        super(EncoderFeedForwardNetwork, self).__init__()

        self.recurrent1 = encoder_recurrent.EncoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.recurrent2 = encoder_recurrent.EncoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.recurrent3 = encoder_recurrent.EncoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.recurrent4 = encoder_recurrent.EncoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
        self.recurrent5 = encoder_recurrent.EncoderRecurrentLayer(input_size=2, hidden_size=3, output_size=2)
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