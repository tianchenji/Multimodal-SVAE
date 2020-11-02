import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, layer_sizes):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+2 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, x):

        # preprocess input size
        assert x.dim() <= 2

        return self.MLP(x)