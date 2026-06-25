import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ── Building blocks ──────────────────────────────────────────────────────────


def _conv_down(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _conv_up(in_ch: int, out_ch: int) -> nn.Sequential:
    # GroupNorm only — ReLU is applied after FiLM so the condition modulates before activation
    num_groups = min(32, max(1, out_ch // 8))
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.GroupNorm(num_groups, out_ch),
    )


class SelfAttn2d(nn.Module):
    """Residual self-attention at a given spatial resolution."""

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch // 8), ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # (B, H*W, C)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).view(B, C, H, W)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: c → per-channel gamma/beta applied to conv features."""

    def __init__(self, num_attrs: int, num_ch: int):
        super().__init__()
        self.gamma = nn.Linear(num_attrs, num_ch)
        self.beta  = nn.Linear(num_attrs, num_ch)
        # Init so the layer starts as an identity transform
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        γ = self.gamma(c)[:, :, None, None]  # (B, C, 1, 1)
        β = self.beta(c)[:, :, None, None]
        return γ * x + β


# ── Encoder ───────────────────────────────────────────────────────────────────


class Encoder(nn.Module):
    """Image + condition → (μ, log σ²).

    CNN path: 3×64 → 64×32 → 128×16 → 256×8 → [self-attn] → 512×4
    Then flatten, concat condition, project to μ and logvar.
    """

    def __init__(self, img_size: int = 64, num_attrs: int = 40, latent_dim: int = 256):
        super().__init__()
        self.down1 = _conv_down(3, 64)
        self.down2 = _conv_down(64, 128)
        self.down3 = _conv_down(128, 256)
        self.attn  = SelfAttn2d(256, num_heads=4)   # at 8×8 spatial resolution
        self.down4 = _conv_down(256, 512)

        spatial = img_size // 16
        flat_dim = 512 * spatial * spatial  # 8192 for img_size=64

        self.fc = nn.Sequential(
            nn.Linear(flat_dim + num_attrs, 1024),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.down4(self.attn(self.down3(self.down2(self.down1(x))))).flatten(1)
        h = self.fc(torch.cat([h, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


# ── Decoder ───────────────────────────────────────────────────────────────────


class Decoder(nn.Module):
    """(z, condition) → image in [-1, 1].

    FiLM layers after each deconv block inject the condition at every scale,
    forcing the decoder to honour c rather than relying solely on z.
    """

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

        chs = [512, 256, 128, 64, 32]
        self.ups   = nn.ModuleList([_conv_up(chs[i], chs[i + 1]) for i in range(4)])
        self.films = nn.ModuleList([FiLM(num_attrs, chs[i + 1]) for i in range(4)])
        self.attn  = SelfAttn2d(256, num_heads=4)  # at 8×8, after first up (512→256)

        self.head = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(h.size(0), 512, self.spatial, self.spatial)
        for i, (up, film) in enumerate(zip(self.ups, self.films)):
            h = F.relu(film(up(h), c), inplace=True)
            if i == 0:
                h = self.attn(h)  # self-attention at 8×8 (after 512→256 up)
        return self.head(h)


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
        """Encode with source condition, decode with target condition."""
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
