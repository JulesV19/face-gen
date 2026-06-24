"""
Gaussian diffusion (DDPM forward process, DDIM sampling).

The model predicts the noise ε. Unlike the VAE's pixel MSE, this MSE-on-noise
objective does NOT cause blur: generation is an iterative denoising chain, not a
one-shot reconstruction, so high-frequency detail is preserved.

Classifier-free guidance: during training the condition is randomly dropped; at
sampling we mix the conditional and unconditional ε predictions to steer the
result towards the requested attributes.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule — better than linear at low resolution."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    acp = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - acp[1:] / acp[:-1]
    return betas.clamp(1e-8, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)


class GaussianDiffusion(nn.Module):
    """Holds the noise schedule as buffers and implements train loss + sampling."""

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.timesteps = timesteps
        betas = (
            cosine_beta_schedule(timesteps)
            if schedule == "cosine"
            else linear_beta_schedule(timesteps)
        )
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))
        self.register_buffer("sqrt_recip_acp", torch.sqrt(1.0 / acp))
        self.register_buffer("sqrt_recipm1_acp", torch.sqrt(1.0 / acp - 1.0))

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε."""
        return (
            self._extract(self.sqrt_acp, t, x0.shape) * x0
            + self._extract(self.sqrt_one_minus_acp, t, x0.shape) * noise
        )

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_acp, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_acp, t, x_t.shape) * eps
        )

    # ── training ──────────────────────────────────────────────────────────

    def p_losses(self, model, x0: torch.Tensor, attrs: torch.Tensor, cond_drop_prob: float) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        drop = torch.rand(b, device=x0.device) < cond_drop_prob
        pred = model(x_t, t, attrs, cond_drop_mask=drop)
        return F.mse_loss(pred, noise)

    # ── sampling (DDIM + classifier-free guidance) ─────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        attrs: torch.Tensor,
        guidance_scale: float = 3.0,
        ddim_steps: int = 50,
        eta: float = 0.0,
        img_size: int = 64,
    ) -> torch.Tensor:
        """Sample images conditioned on `attrs` (B, num_attrs). Returns (B,3,H,W) in [-1,1]."""
        device = attrs.device
        b = attrs.size(0)
        x = torch.randn(b, 3, img_size, img_size, device=device)
        times = torch.linspace(self.timesteps - 1, 0, ddim_steps, device=device).long()

        uncond = torch.ones(b, dtype=torch.bool, device=device)
        cond = torch.zeros(b, dtype=torch.bool, device=device)

        for i in range(ddim_steps):
            t = times[i]
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            # One batched forward pass for both conditional and unconditional ε.
            eps_cond, eps_uncond = model(
                torch.cat([x, x]),
                torch.cat([t_batch, t_batch]),
                torch.cat([attrs, attrs]),
                cond_drop_mask=torch.cat([cond, uncond]),
            ).chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            x0 = self.predict_x0(x, t_batch, eps).clamp(-1, 1)

            acp_t = self.alphas_cumprod[t]
            if i < ddim_steps - 1:
                acp_prev = self.alphas_cumprod[times[i + 1]]
            else:
                acp_prev = torch.ones((), device=device)

            sigma = eta * torch.sqrt(
                (1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev)
            )
            dir_xt = torch.sqrt((1 - acp_prev - sigma**2).clamp(min=0)) * eps
            x = torch.sqrt(acp_prev) * x0 + dir_xt
            if eta > 0 and i < ddim_steps - 1:
                x = x + sigma * torch.randn_like(x)

        return x.clamp(-1, 1)
