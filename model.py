import torch
import torch.nn as nn
from typing import Tuple


# ── Building blocks ──────────────────────────────────────────────────────────


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


# ── Encoder ───────────────────────────────────────────────────────────────────


class Encoder(nn.Module):
    """Image + condition → (μ, log σ²).

    CNN path: 3×64 → 64×32 → 128×16 → 256×8 → 512×4
    Then flatten, concat condition, project to μ and logvar.
    """

    def __init__(self, img_size: int = 64, num_attrs: int = 40, latent_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_down(3, 64),
            _conv_down(64, 128),
            _conv_down(128, 256),
            _conv_down(256, 512),
        )
        # spatial size after 4× stride-2: img_size // 16
        spatial = img_size // 16
        flat_dim = 512 * spatial * spatial  # 8192 for img_size=64

        self.fc = nn.Sequential(
            nn.Linear(flat_dim + num_attrs, 1024),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.cnn(x).flatten(1)
        h = self.fc(torch.cat([h, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


# ── Decoder ───────────────────────────────────────────────────────────────────


class Decoder(nn.Module):
    """(z, condition) → image in [-1, 1]."""

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

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(h.size(0), 512, self.spatial, self.spatial)
        return self.deconv(h)


# ── CVAE ─────────────────────────────────────────────────────────────────────


class CVAE(nn.Module):
    def __init__(self, img_size: int = 64, num_attrs: int = 40, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(img_size, num_attrs, latent_dim)
        self.decoder = Decoder(img_size, num_attrs, latent_dim)

    # ── core ──────────────────────────────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, c)
        # Clamp logvar so exp() (in reparam std and in the KL term) cannot
        # overflow to inf — without this the reparameterisation explodes during
        # the β=0 warmup epoch (no KL pressure on the variance) and everything
        # turns into NaN. Range [-10, 10] keeps std in [~0.007, ~148].
        logvar = logvar.clamp(-10.0, 10.0)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar

    # ── inference helpers ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, c: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Sample z ~ N(0,I) and decode with condition c.

        c: (num_attrs,) or (n, num_attrs)
        """
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(n, -1)
        z = torch.randn(c.size(0), self.latent_dim, device=c.device, dtype=c.dtype)
        return self.decoder(z, c)

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Deterministic encode — returns μ (no sampling)."""
        mu, _ = self.encoder(x, c)
        return mu

    @torch.no_grad()
    def manipulate(
        self,
        x: torch.Tensor,
        c_source: torch.Tensor,
        c_target: torch.Tensor,
    ) -> torch.Tensor:
        """Encode with source condition, decode with target condition.

        Lets you flip one or more attributes on a real photo.
        """
        mu = self.encode_mu(x, c_source)
        return self.decoder(mu, c_target)


# ── Auxiliary attribute classifier ────────────────────────────────────────────


class AttrClassifier(nn.Module):
    """Small CNN: image in [-1, 1] → 40 attribute logits.

    Optional auxiliary signal (enabled by Config.attr_loss_weight > 0): trained on
    real images, then used to push the CVAE decoder into actually honouring the
    condition c — counters the classic failure where the decoder ignores c because
    z already carries every attribute it needs to reconstruct.
    """

    def __init__(self, img_size: int = 64, num_attrs: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            _conv_down(3, 64),
            _conv_down(64, 128),
            _conv_down(128, 256),
            _conv_down(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, num_attrs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))
