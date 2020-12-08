from typing import Dict

import torch
from torch import nn, optim, Tensor
from torchvision.models.resnet import resnet50


class Embeddings(nn.Module):
    """
    Creates representation of the frame and history as a single vector (context embedding).
    """
    def __init__(self, cfg: Dict):
        super().__init__()

        num_history_channels = (cfg['model_params']['history_num_frames'] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone = resnet50(pretrained=True)
        self.backbone_n_out = 2048
        self.n_head = cfg['embed_params']['n_head']
        self.emb_dim = cfg['embed_params']['emb_dim']

        # Adjust input channel for the Lyft data
        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False
        )

        self.embeddings = nn.Sequential(
            nn.Linear(in_features=self.backbone_n_out, out_features=self.n_head),
            nn.ReLU(),
            nn.Linear(self.n_head, self.emb_dim)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        embeddings = self.embeddings(x)

        return embeddings


class Encoder(nn.Module):
    """
    Encoder of conditional variational auto-encoder.
    """
    def __init__(self, cfg: Dict):
        super().__init__()

        self.latent_dim = cfg['cvae_cfg']['latent_dim']
        self.frame_embedding_dim = cfg['embed_params']['emb_dim']
        self.trajectory_length = cfg['model_params']['future_num_frames'] * 2
        layers_dims = [self.trajectory_length + self.frame_embedding_dim] + cfg['cvae_cfg']['encoder_layers']

        self.layers = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layers_dims[:-1], layers_dims[1:])):
            self.layers.add_module(name=f'layer{i}', module=nn.Linear(in_size, out_size))
            self.layers.add_module(name=f'relu{i}', module=nn.ReLU())

        self.linear_means = nn.Linear(in_features=layers_dims[-1], out_features=self.latent_dim)
        self.linear_log_var = nn.Linear(in_features=layers_dims[-1], out_features=self.latent_dim)

    def forward(self, x, emb):
        """
        Args:
            x (Tensor): ground truth future trajectory, shape: [bs, time_steps * 2] (stacked x and y coordinates).
            emb (Tensor): embedding of frame and history for conditioning, shape: [bs, emb_dim].
        Returns:
            means (Tensor): mean of distribution P(trajectory | embedding), shape: [bs, latent_dim].
            log_var (Tensor): log of distribution variance, shape: [bs, latent_dim].
        """
        x = torch.cat((x, emb), dim=1)
        x = self.layers(x)
        means = self.linear_means(x)
        log_var = self.linear_log_var(x)
        return means, log_var


class Decoder(nn.Module):
    """
    Decoder of conditional variational auto-encoder.
    """
    def __init__(self, cfg: Dict):
        super().__init__()

        self.trajectory_length = cfg['model_params']['future_num_frames'] * 2
        layers = [cfg['cvae_cfg']['latent_dim'] + cfg['embed_params']['emb_dim']] + cfg['cvae_cfg']['decoder_layers']

        self.layers = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.layers.add_module(name=f'layer{i}', module=nn.Linear(in_size, out_size))
            self.layers.add_module(name=f'relu{i}', module=nn.ReLU())

        self.reconstruction = nn.Linear(layers[-1], self.trajectory_length)

    def forward(self, x, emb):
        """
        Args:
            x (Tensor): vector from the latent distribution, shape: [bs, latent_dim].
            emb (Tensor): embedding of frame and history for conditioning, shape: [bs, emb_dim].
        Returns:
            x (Tensor): reconstructed future coordinates, shape [bs, time_steps * 2] (stacked x and y coordinates).
        """
        x = torch.cat((x, emb), dim=1)
        x = self.layers(x)
        x = self.reconstruction(x)
        return x


class CVAE(nn.Module):
    """
    Conditional variational auto-encoder.
    Perform future trajectory auto-encoding conditioned on frame and history embedding.
    Learns distribution P(trajectory | embedding).
    """
    def __init__(self, cfg: Dict):
        super().__init__()

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.embeddings = Embeddings(cfg)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, context):
        c = self.embeddings(context)
        recon = self.decoder(z, c)
        return recon

    def forward(self, future_trj, context):
        c = self.embeddings(context)
        means, log_var = self.encoder(future_trj, c)
        z = self.reparametrize(means, log_var)
        recon = self.decoder(z, c)
        return recon, means, log_var, z


class TrajectoriesExtractor(nn.Module):
    """
    Extract 3 trajectory from samples generated by CVAE.
    """
    def __init__(self, cfg: Dict):
        super().__init__()

        n_samples = cfg['extractor_cfg']['n_samples']
        n_head = cfg['extractor_cfg']['n_head']
        p_drop = cfg['extractor_cfg']['p_drop']
        n_out = cfg['model_params']['future_num_frames'] * 2
        self.n_channels = cfg['extractor_cfg']['n_channels']

        self.extractor = nn.Sequential(
            nn.Conv2d(1, self.n_channels, (n_samples, 1), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(self.n_channels, 1, (1, 1), (1, 1), (0, 0)),
            nn.ReLU())

        self.head1 = nn.Sequential(
            nn.Linear(200, n_head),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(n_head, n_out))
        self.head2 = nn.Sequential(
            nn.Linear(200, n_head),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(n_head, n_out))
        self.head3 = nn.Sequential(
            nn.Linear(200, n_head),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(n_head, n_out))

    def forward(self, x):
        """
        Args:
            x (Tensor): samples of future trajectories from CVAE, shape: [bs, 1, n_samples, n_time_steps * 2].
        Returns:
            tr (Tensor): extracted future trajectories, shape [bs, 3, n_time_steps, 2].
        """
        f = self.extractor(x)
        f = torch.flatten(f, 1)
        x_mean = torch.mean(x, 2).view(-1, 100)
        f = torch.cat((f, x_mean), 1)

        tr_1 = self.head1(f)
        tr_2 = self.head2(f)
        tr_3 = self.head3(f)

        tr = torch.cat((tr_1, tr_2, tr_3), dim=1)
        tr = tr.view(-1, 3, 50, 2)
        return tr


class TrajectoriesPredictor(nn.Module):
    def __init__(self, cvae_model, extractor_model, cfg, device):
        super().__init__()
        self.cvae_model = cvae_model
        self.extractor_model = extractor_model
        self.cfg = cfg
        self.device = device

    def sample_trajectories_batch(self, context):
        """
        Samples trajectories form CVAE given context.
        Returns:
            samples (Tensor): batch of samples given context, shape: [bs, 1, n_samples, 2* n_time_steps].
        """
        n_samples = self.cfg['extractor_cfg']['n_samples']
        n_time_steps = self.cfg['model_params']['future_num_frames']
        bs = context.shape[0]
        samples = torch.zeros((bs, 1, n_samples, 2 * n_time_steps))
        for i in range(n_samples):
            z = torch.randn(bs, cfg['cvae_cfg']['latent_dim']).to(self.device)
            with torch.no_grad():
                trajectories = self.cvae_model.inference(z, context)
            samples[:, 0, i, :] = trajectories
        return samples

    def forward(self, x):
        context = x['image'].to(self.device)
        trajectories = self.sample_trajectories_batch(context).to(self.device)
        predicitons = self.extractor_model(trajectories)
        return predicitons
