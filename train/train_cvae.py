import torch
import numpy as np

from tqdm import tqdm
from IPython.display import clear_output

from utils import cvae_training_monitor
from train.losses import loss_KLD, loss_MMD, compute_kernel, compute_mmd


def forward_cvae(data, model, device, cfg):
    """
    Compute forward pass for CVAE model.
    Supports 2 types of VLB loss: KLD and MMD.
    Returns:
        recon (Tensor): reconstructed trajectory.
        rec_loss (Tensor): reconstruction loss.
        vlb_loss: variational loss.
    """
    criterion = cfg['cvae_cfg']['vlb_loss']
    context = data['image'].to(device)
    targets = data['target_positions']  # [bs, 50, 2]
    targets_xy = torch.cat((targets[:, :, 0], targets[:, :, 1]), dim=1).to(device)  # [bs, 100], first 50 - x, next - y

    # Forward pass
    recon, means, log_var, z = model(targets_xy, context)
    rec_loss = torch.mean(torch.mean((recon - targets_xy).pow(2), dim=1))
    if criterion == 'KLD':
        vlb_loss = loss_KLD(means, log_var, cfg)
    if criterion == 'MMD':
        vlb_loss = loss_MMD(z)
    return recon, rec_loss, vlb_loss


def train_cvae(model, data_loader, optimizer, device, cfg,
               plot_mode=True):

    checkpoint_path = cfg['models_checkpoint_path']
    criterion = cfg['cvae_cfg']['vlb_loss']

    tr_it = iter(data_loader)
    progress_bar = tqdm(range(cfg['train_cvae_params']['max_num_steps']))

    losses_train = []
    recon_losses = []
    vlb_losses = []

    for i in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(data_loader)
            data = next(tr_it)

        model.train()
        torch.set_grad_enabled(True)
        # Forward
        _, recon_loss, vlb_loss = forward_cvae(data, model, device, cfg)
        loss = recon_loss + vlb_loss
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item()) # mean per batch
        recon_losses.append(recon_loss.item())
        vlb_losses.append(vlb_loss.item())
        progress_bar.set_description(f'loss: {loss.item()}, loss(avg): {np.mean(losses_train)}')
        clear_output(True)
        if plot_mode:
            cvae_training_monitor(recon_losses, vlb_losses, criterion)

        if i % cfg['train_cvae_params']['checkpoint_every_n_steps'] == 0 and i > 0:
            mean_loss = np.mean(losses_train)
            torch.save({
                  'iteration': i,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': mean_loss},
                 f'{checkpoint_path}/{model.__class__.__name__}_{i}_{mean_loss:.0f}')
