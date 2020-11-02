import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):

        x = self.MLP(z)

        return x