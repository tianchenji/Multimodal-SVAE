import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.SVAE import SVAE
from custom_dataset import TerraDataset
from utils.print_statistics import print_statistics

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

    #-------------------------------------create dataset---------------------------------------
    # replace this part with your own dataset

    clip_thres = 1850
    dataset    = TerraDataset(data_root='test_set/', clip_thres=clip_thres, test_flag=1)
    #------------------------------------------------------------------------------------------

    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False)

    print("Dataset is ready")

    svae = SVAE(
        device='cpu',
        dim_x_l=6,
        dim_x_h=1080,
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        classifier_layer_sizes=args.classifier_layer_sizes)

    PATH = './net_weights/terra_net_svae.pth'
    svae.load_state_dict(torch.load(PATH))

    # start testing
    #---------------------------------simplified statistics------------------------------------
    # compute the overall accuracy of the classification on the test set

    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            x1, x2, x3, y = data
            _, _, _, _, pred_labels_score = svae(x1, torch.cat((x2, x3), dim=-1))
            _, pred_labels = torch.max(pred_labels_score, 1)
            total += y.size(0)
            correct += (pred_labels == y).sum().item()

    print('Accuracy of the network over the test set: %f %%' % (100 * correct / total))

    #------------------------------------------------------------------------------------------

    '''
    #----------------------------------detailed statistics-------------------------------------
    # compute precisions for each label, Kappa coefficient, and confusion matrix on the test set
    # Modifications for this segment and utils.print_statistics may needed if you have different
    # labels in your problem.

    correct     = [0] * 4
    confusion_m = [[0]*4 for _ in range(4)]
    total       = [0] * 4
    with torch.no_grad():
        for data in data_loader:
            x1, x2, x3, y = data
            _, _, _, _, pred_labels_score = svae(x1, torch.cat((x2, x3), dim=-1))
            _, pred_labels = torch.max(pred_labels_score, 1)
            for i in range(4):
                total[i] += (y == i).sum().item()
                correct[i] += ((pred_labels == y) * (y == i)).sum().item()
                for j in range(4):
                    confusion_m[j][i] += ((pred_labels == j) * (y == i)).sum().item()

    print_statistics(
        correct, confusion_m, total, args.confusion_matrix)

    #-------------------------------------------------------------------------------------------
    '''

    '''
    #----------------------visualization of the latent space interpretability-------------------
    # generate Figure 5(a) in the paper

    seg_size = 10
    x = svae.inference(n=seg_size)
    theta = np.arange(-0.25*np.pi, 1.25*np.pi, np.pi/720)
    plt.figure(figsize=(10, 10))
    for p in range(seg_size * seg_size):
        plt.subplot(seg_size, seg_size, p+1)
        dist = x[p].data.numpy()
        cur_x = dist * np.cos(theta)
        cur_y = dist * np.sin(theta)
        plt.plot(cur_x, cur_y, ls='None', marker='.', markersize=1)
        plt.axis([-0.5, 0.5, -0.5, 0.5])
        plt.grid(True)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)

    plt.show()

    #-------------------------------------------------------------------------------------------
    '''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1080, 128])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[128, 1080])
    parser.add_argument("--classifier_layer_sizes", type=list, default=[64, 4])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--confusion_matrix", action='store_true')

    args = parser.parse_args()

    main(args)