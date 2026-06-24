"""
Inference for the conditional diffusion model — sample faces with chosen attributes.

    python generate_diffusion.py \\
        --ckpt checkpoints_diffusion/best.ckpt \\
        --attrs "Blond_Hair=1,Smiling=1,Young=1" \\
        --n 8 --guidance 3.0 --out generated_diffusion.png

Higher --guidance pushes harder towards the requested attributes (typical 1-5);
too high can over-saturate or reduce diversity.
"""
import argparse

import torch

from config import DiffusionConfig
from diffusion import GaussianDiffusion
from generate import attrs_from_str, build_condition, save_grid
from unet import UNet


def load_model(ckpt_path: str, device: str = "cuda") -> tuple[UNet, GaussianDiffusion, DiffusionConfig]:
    """Load a Lightning checkpoint into the bare UNet (no pytorch_lightning needed).

    Uses the EMA weights — that is what produces clean samples.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = DiffusionConfig(**ckpt["hyper_parameters"])
    unet = UNet(
        img_size=cfg.img_size,
        num_attrs=cfg.num_attrs,
        base_channels=cfg.base_channels,
        channel_mults=tuple(cfg.channel_mults),
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=tuple(cfg.attn_resolutions),
        dropout=cfg.dropout,
    )
    state = {
        k[len("ema_unet."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("ema_unet.")
    }
    unet.load_state_dict(state)
    unet.eval().to(device)

    diffusion = GaussianDiffusion(cfg.timesteps, cfg.schedule).to(device)
    return unet, diffusion, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attrs", default="", help="e.g. 'Smiling=1,Blond_Hair=1'")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    p.add_argument("--out", default="generated_diffusion.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet, diffusion, cfg = load_model(args.ckpt, device)

    overrides = attrs_from_str(args.attrs) if args.attrs else {}
    c = build_condition(overrides, cfg.num_attrs, default=0.0, device=device)
    attrs = c.unsqueeze(0).expand(args.n, -1).contiguous()

    samples = diffusion.ddim_sample(
        unet,
        attrs,
        guidance_scale=args.guidance,
        ddim_steps=args.steps,
        img_size=cfg.img_size,
    )
    save_grid(samples, args.out, nrow=args.n)


if __name__ == "__main__":
    main()
