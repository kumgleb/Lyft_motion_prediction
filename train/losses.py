import torch
import numpy as np


def loss_KLD(mean, log_var, cfg, batch_mean=True):
    """
    Computes Kullback–Leibler divergence for Gaussian prior.
    Optional: betta < 1 allows encoder distribution to diverge more form the prior (standard Gaussian).
    """
    betta = cfg['cvae_cfg']['betta']
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    if batch_mean:
        bs = mean.shape[0]
        KLD = KLD / bs
    return betta * KLD / bs


def loss_MMD(z, cfg):
    """
    Compute maximum mean discrepancy loss between encoder and prior distributions.
    source: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    """
    betta = cfg['cvae_cfg']['betta']
    z_dim = cfg['cvae_cfg']['latent_dim']
    prior_samples = torch.randn(200, z_dim)
    MMD = compute_mmd(prior_samples, z)
    return MMD * betta


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


def neg_multi_log_likelihood_batch(gt, pred, confidences, avails):
    """
    Compute negative log likelihood for multi-modal trajectory prediction with log-sum-exp trick.
    """
    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability
    with np.errstate(divide="ignore"):
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value
    return torch.mean(error)
