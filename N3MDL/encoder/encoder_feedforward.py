import torch.nn as nn
from activation import relu
from layers import encoder_recurrent

class EncoderFeedForwardNetwork(nn.Module):
    def __init__(self, model_dimensionality: int, hidden_dimensionality: int, feedforward_dimensionality: int):
        super(EncoderFeedForwardNetwork, self).__init__()

        self.recurrent1 = encoder_recurrent.EncoderRecurrentLayer(input_size=model_dimensionality, hidden_size=hidden_dimensionality,
                                                                  output_size=feedforward_dimensionality)
        self.recurrent2 = encoder_recurrent.EncoderRecurrentLayer(input_size=model_dimensionality, hidden_size=hidden_dimensionality,
                                                                  output_size=feedforward_dimensionality)
        self.recurrent3 = encoder_recurrent.EncoderRecurrentLayer(input_size=model_dimensionality, hidden_size=hidden_dimensionality,
                                                                  output_size=feedforward_dimensionality)
        self.recurrent4 = encoder_recurrent.EncoderRecurrentLayer(input_size=model_dimensionality, hidden_size=hidden_dimensionality,
                                                                  output_size=feedforward_dimensionality)
        self.recurrent5 = encoder_recurrent.EncoderRecurrentLayer(input_size=model_dimensionality, hidden_size=hidden_dimensionality,
                                                                  output_size=feedforward_dimensionality)
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