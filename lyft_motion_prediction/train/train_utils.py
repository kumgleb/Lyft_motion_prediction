import torch


def sample_trajectories_batch(model, context, device, cfg):
    """
    Samples trajectories form CVAE given context.
    Returns:
        samples (Tensor): batch of samples given context, shape: [bs, 1, n_samples, 2* n_time_steps].
    """
    n_samples = cfg['extractor_cfg']['n_samples']
    n_time_steps = cfg['model_params']['future_num_frames']
    bs = context.shape[0]
    samples = torch.zeros((bs, 1, n_samples, 2 * n_time_steps))
    for i in range(n_samples):
        z = torch.randn(bs, cfg['cvae_cfg']['latent_dim']).to(device)
        with torch.no_grad():
            trajectories = model.inference(z, context)
        samples[:, 0, i, :] = trajectories
    return samples



