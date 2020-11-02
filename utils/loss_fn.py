import torch

num_points    = 1080
no_loss_thres = 1800

def loss_fn_SVAE(recon_x, x, mean, log_var, pred_labels_score, y, alpha):

    '''
    # ignore loss of remote lidar points
    x = x.to('cpu')

    assert len(x.size()) == 2 

    num_rows, num_columns = x.size()
    mask_tensor = torch.zeros(num_rows, num_columns)

    for row_idx in range(num_rows):
        for column_idx in range(num_columns):
            if x[row_idx][column_idx] * clip_thres <= no_loss_thres:
                mask_tensor[row_idx][column_idx] = 1.0

    x = x.to(device)

    mask_tensor = mask_tensor.to(device)
    recon_x = recon_x * mask_tensor
    x = x * mask_tensor
    '''

    BCE = torch.nn.functional.mse_loss(
        recon_x.view(-1, num_points), x.view(-1, num_points), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    CLF = torch.nn.functional.cross_entropy(pred_labels_score, y, reduction='sum')

    return (BCE + KLD + alpha * CLF) / x.size(0)

def loss_fn_generative(recon_x, x, mean, log_var):

    BCE = torch.nn.functional.mse_loss(
        recon_x.view(-1, num_points), x.view(-1, num_points), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)

def loss_fn_discriminative(pred_labels_score, y):

    CLF = torch.nn.functional.cross_entropy(pred_labels_score, y, reduction='sum')

    return CLF / y.size(0)