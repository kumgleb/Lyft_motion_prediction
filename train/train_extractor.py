import torch
import numpy as np

from tqdm import tqdm
from IPython.display import clear_output

from utils import extractor_training_monitor
from train.train_utils import sample_trajectories_batch
from train.losses import neg_multi_log_likelihood_batch


def forward_extractor(cvae_model, extractor_model, data, device, criterion, confs, cfg):
    """
    Compute forward pass for Extractor model.
    Returns:
        preds (Tensor): reconstructed trajectory.
        loss (Tensor): loss w.r.t. criterion.
    """
    context = data['image'].to(device)
    trajectories = sample_trajectories_batch(cvae_model, context, device, cfg).to(device)
    target_availabilities = data['target_availabilities'].to(device)
    targets = data['target_positions'].to(device)
    # Forward pass
    preds = extractor_model(trajectories)
    loss = criterion(targets, preds, confs, target_availabilities)
    return loss, preds


def train_extractor(cvae_model, extractor_model,
                    data_loader, confs,
                    optimizer, device, cfg,
                    plot_mode=True):

    checkpoint_path = cfg['models_checkpoint_path']

    tr_it = iter(data_loader)
    progress_bar = tqdm(range(cfg['train_extractor_params']['max_num_steps']))

    losses_train = []
    iterations = []

    for i in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)

        extractor_model.train()
        torch.set_grad_enabled(True)
        # Forward
        loss, preds = forward_extractor(cvae_model, extractor_model, data, device,
                                        neg_multi_log_likelihood_batch, confs, cfg)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iterations.append(i)
        losses_train.append(loss.item())  # mean per batch
        progress_bar.set_description(f'loss: {loss.item()}, loss(avg): {np.mean(losses_train)}')
        clear_output(True)
        if plot_mode:
            extractor_training_monitor(losses_train)

        if i % cfg['train_extractor_params']['checkpoint_every_n_steps'] == 0 and i > 0:
            mean_loss = np.mean(losses_train)
            torch.save({
                  'iteration': i,
                  'model_state_dict': extractor_model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': mean_loss},
                 f'{checkpoint_path}/{extractor_model.__class__.__name__}_{i}_{mean_loss:.0f}')
