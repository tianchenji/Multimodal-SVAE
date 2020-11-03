# Multimodal Supervised Variational Autoencoder (SVAE)
This repository stores the Pytorch implementation of the SVAE for the following paper:

**T. Ji, S. T. Vuppala, G. Chowdhary and K. Driggs-Campbell, "Multi-Modal Anomaly Detection for Unstructured and Uncertain Environments", in *Conference on Robot Learning (CoRL)*, 2020**

Generally, SVAEs can be applied to supervised learning problems where the input consists of a high-dimensional modality *x_h* and a low-dimensional modality *x_l*. SVAEs can also be used in the context of unimodal inputs by removing *x_l* from the classifier input layer. See the paper for more details.

Part of the code was adapted from [T. Baumgärtner. VAE-CVAE-MNIST.](https://github.com/timbmg/VAE-CVAE-MNIST)

## Prerequisites
1. [PyTorch](https://pytorch.org/get-started/locally/)
2. (Optional) CUDA GPU support

Tested using Python 3.7 and Pytorch 1.5.0.

## Dataset
The field robot data used in our experiments are provided by [EarthSense](https://www.earthsense.co/). At this time, we may not be able to open source the dataset due to copyright issues. As a result, this repository does not aim to reproduce the results in the paper, but to **provide a Pytorch framework of SVAEs in a more general setting**.

## Description of the code
More detailed comments can be found in the code. Here are some general descriptions:
* `models`: Contains network architectures for the SVAE.

* `net_weights`: Contains trained network parameters.

* `utils`: Contains the code for loss functions and metrics for quantitative results.

* `custom_dataset.py`: Loads the dataset. One should modify this file to create a customized dataset for their own problems.

* `train.py`: Train the SVAE.

* `test.py`: Test the SVAE. To get the confusion matrix for the SVAE, uncomment `detailed statistics` part in the code and add `--confusion_matrix` to the command.

## Contact
Should you have any questions or comments on the code, please feel free to open an issue or contact the author at tj12@illinois.edu.
