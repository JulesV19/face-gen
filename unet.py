"""
Conditional UNet for diffusion on CelebA (64×64, 40 binary attributes).

Time step and attribute condition are embedded, summed, and injected into every
residual block (FiLM-style additive). A learned `null_emb` replaces the attribute
embedding when the condition is dropped — this is what makes classifier-free
guidance possible at sampling time.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── embeddings ──────────────────────────────────────────────────────────────


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of a (B,) tensor of integer timesteps → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _norm(ch: int) -> nn.GroupNorm:
    # All channel counts here are multiples of 32 (base=128 × mults).
    return nn.GroupNorm(32, ch)


# ── blocks ──────────────────────────────────────────────────────────────────


class ResBlock(nn.Module):
    """GroupNorm → SiLU → conv, with the time/attr embedding added in the middle."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial positions (used at low resolutions)."""

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = _norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w).unbind(1)
        scale = (c // self.num_heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class TimestepBlock(nn.Module):
    """Sequential wrapper that feeds `emb` to ResBlocks and only `x` to the rest."""

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, emb):
        for layer in self.layers:
            x = layer(x, emb) if isinstance(layer, ResBlock) else layer(x)
        return x


# ── UNet ────────────────────────────────────────────────────────────────────


class UNet(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        in_ch: int = 3,
        num_attrs: int = 40,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_channels = base_channels
        emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.attr_mlp = nn.Sequential(
            nn.Linear(num_attrs, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        # Replaces the attribute embedding when the condition is dropped (CFG).
        self.null_emb = nn.Parameter(torch.zeros(emb_dim))

        self.in_conv = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        # ── down path ─────────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        skip_chs = [base_channels]
        ch = base_channels
        res = img_size
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, emb_dim, dropout)]
                ch = out_ch
                if res in attn_resolutions:
                    layers.append(AttentionBlock(ch))
                self.down_blocks.append(TimestepBlock(layers))
                skip_chs.append(ch)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(TimestepBlock([Downsample(ch)]))
                skip_chs.append(ch)
                res //= 2

        # ── middle ────────────────────────────────────────────────────────
        self.mid = TimestepBlock(
            [
                ResBlock(ch, ch, emb_dim, dropout),
                AttentionBlock(ch),
                ResBlock(ch, ch, emb_dim, dropout),
            ]
        )

        # ── up path ───────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for j in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skip_chs.pop(), out_ch, emb_dim, dropout)]
                ch = out_ch
                if res in attn_resolutions:
                    layers.append(AttentionBlock(ch))
                if i != 0 and j == num_res_blocks:
                    layers.append(Upsample(ch))
                    res *= 2
                self.up_blocks.append(TimestepBlock(layers))

        self.out = nn.Sequential(_norm(ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attrs: torch.Tensor,
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B,3,H,W), t: (B,), attrs: (B,num_attrs), cond_drop_mask: (B,) bool."""
        emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        a = self.attr_mlp(attrs)
        if cond_drop_mask is not None:
            a = torch.where(cond_drop_mask[:, None], self.null_emb[None].to(a.dtype), a)
        emb = emb + a

        h = self.in_conv(x)
        hs = [h]
        for block in self.down_blocks:
            h = block(h, emb)
            hs.append(h)
        h = self.mid(h, emb)
        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, emb)
        return self.out(h)
