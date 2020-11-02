import sys
import torch
import argparse
from torch.utils.data import DataLoader

from models.SVAE import SVAE
from custom_dataset import TerraDataset
from utils.loss_fn import loss_fn_SVAE

def main(args):
    '''
    SVAE
    --------------------------
    
    Inputs:  x_h - high-dimensional data (dimension of dim_x_h)
             x_l - low-dimensional data (dimension of dim_x_l)

    Outputs: pred_labels - inferred labels

    Usage: svae(x_h, x_l)
    '''

    # RNG is a sequential process
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #-------------------------------------create dataset---------------------------------------
    # replace this part with your own dataset

    clip_thres = 1850
    dataset    = TerraDataset(data_root='training_set/', clip_thres=clip_thres, test_flag=0)
    #------------------------------------------------------------------------------------------

    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    print("Dataset is ready")

    # relative weight between generative and purely discriminative learning
    alpha = 0.01 * len(data_loader)

    svae = SVAE(
        device=device,
        dim_x_l=6,
        dim_x_h=1080,
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        classifier_layer_sizes=args.classifier_layer_sizes).to(device)

    optimizer = torch.optim.Adam(svae.parameters(), lr=args.learning_rate)

    # start training
    for epoch in range(args.epochs):

        for iteration, (x1, x2, x3, y) in enumerate(data_loader):

            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            # x1 is x_h, (x2, x3) is x_l
            recon_x1, mean, log_var, z, pred_labels_score = svae(x1, torch.cat((x2, x3), dim=-1))

            loss = loss_fn_SVAE(recon_x1, x1, mean, log_var, pred_labels_score, y, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

    PATH = './net_weights/terra_net_svae.pth'
    torch.save(svae.state_dict(), PATH)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1080, 128])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[128, 1080])
    parser.add_argument("--classifier_layer_sizes", type=list, default=[64, 4])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args()

    main(args)