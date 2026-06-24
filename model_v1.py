"""Original CVAE architecture (v1) — no FiLM, no self-attention.

Kept as a frozen copy so the old checkpoint can always be loaded for
comparison, even after model.py has been updated.
"""
import torch
import torch.nn as nn
from typing import Tuple


def _conv_down(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _conv_up(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, img_size: int = 64, num_attrs: int = 40, latent_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_down(3, 64),
            _conv_down(64, 128),
            _conv_down(128, 256),
            _conv_down(256, 512),
        )
        spatial = img_size // 16
        flat_dim = 512 * spatial * spatial
        self.fc = nn.Sequential(
            nn.Linear(flat_dim + num_attrs, 1024),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x, c):
        h = self.cnn(x).flatten(1)
        h = self.fc(torch.cat([h, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, img_size: int = 64, num_attrs: int = 40, latent_dim: int = 256):
        super().__init__()
        self.spatial = img_size // 16
        flat_dim = 512 * self.spatial * self.spatial
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_attrs, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, flat_dim),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            _conv_up(512, 256),
            _conv_up(256, 128),
            _conv_up(128, 64),
            _conv_up(64, 32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, c):
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(h.size(0), 512, self.spatial, self.spatial)
        return self.deconv(h)


class CVAE(nn.Module):
    def __init__(self, img_size: int = 64, num_attrs: int = 40, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(img_size, num_attrs, latent_dim)
        self.decoder = Decoder(img_size, num_attrs, latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        logvar = logvar.clamp(-10.0, 10.0)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar

    @torch.no_grad()
    def generate(self, c: torch.Tensor, n: int = 1) -> torch.Tensor:
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(n, -1)
        z = torch.randn(c.size(0), self.latent_dim, device=c.device, dtype=c.dtype)
        return self.decoder(z, c)
