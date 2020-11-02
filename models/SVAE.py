import torch
import torch.nn as nn

from .blocks.MLP import MLP
from .blocks.Encoder import Encoder
from .blocks.Decoder import Decoder

class SVAE(nn.Module):

    def __init__(self, device, dim_x_l, dim_x_h, encoder_layer_sizes, latent_size,
                 decoder_layer_sizes, classifier_layer_sizes):

        super().__init__()

        self.dim_x_l = dim_x_l
        self.dim_x_h = dim_x_h

        self.device = device

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size)

        latent_para_size = latent_size * 2
        classifier_input_size = latent_para_size + self.dim_x_l

        classifier_layer_sizes = [classifier_input_size] + classifier_layer_sizes

        self.classifier = MLP(classifier_layer_sizes)

    def forward(self, x_h, x_l):

        # flatten the image-like high-dimensional inputs x_h
        if x_h.dim() > 2:
            x_h = x_h.view(-1, self.dim_x_h)

        batch_size = x_h.size(0)

        means, log_var = self.encoder(x_h)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + means

        classifier_inputs = torch.cat((means, log_var, x_l), dim=-1)

        pred_labels_score = self.classifier(classifier_inputs)

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z, pred_labels_score

    def inference(self, n=1):
        '''
        Checking the generative model performance.
        '''

        seg_size = n
        var_1_range = [-2, 2]
        var_2_range = [-2, 2]
        z = torch.empty(0)

        var_1 = torch.linspace(var_1_range[0], var_1_range[1], seg_size).view(seg_size, 1)

        for value in torch.linspace(var_2_range[0], var_2_range[1], seg_size):
            var_2 = torch.tensor([[value]] * seg_size).float()
            z_seg = torch.cat((var_1, var_2), -1)
            z     = torch.cat((z, z_seg), 0)

        recon_x = self.decoder(z)

        return recon_x