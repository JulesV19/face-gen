"""
Inference utilities — generation and attribute manipulation.

Examples
--------
# generate 8 blond smiling young faces
python generate.py generate \\
    --ckpt checkpoints/best.ckpt \\
    --attrs "Blond_Hair=1,Smiling=1,Young=1" \\
    --n 8 --out generated.png

# flip Smiling on a real photo
python generate.py manipulate \\
    --ckpt checkpoints/best.ckpt \\
    --img path/to/face.jpg \\
    --flip "Smiling" \\
    --out manipulated.png
"""
import argparse

import torch
import torchvision
from PIL import Image
from torchvision import transforms

from config import Config
from dataset import ATTR_NAMES
from model import CVAE


# ── helpers ───────────────────────────────────────────────────────────────────


def load_model(ckpt_path: str, device: str = "cuda") -> tuple[CVAE, Config]:
    """Load a Lightning checkpoint into the bare CVAE.

    Inference deliberately does NOT import the training stack (pytorch_lightning,
    wandb): a Lightning .ckpt is just a dict with `state_dict` + `hyper_parameters`.
    We rebuild Config from the saved hparams and strip the `model.` prefix that
    CVAEModule adds (since it holds the net as `self.model`).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["hyper_parameters"])
    model = CVAE(
        img_size=cfg.img_size,
        num_attrs=cfg.num_attrs,
        latent_dim=cfg.latent_dim,
    )
    state = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state)
    model.eval().to(device)
    return model, cfg


def attrs_from_str(spec: str) -> dict[str, float]:
    """Parse 'Smiling=1,Blond_Hair=1' → {'Smiling': 1.0, 'Blond_Hair': 1.0}."""
    result = {}
    for part in spec.split(","):
        name, val = part.strip().split("=")
        result[name.strip()] = float(val.strip())
    return result


def build_condition(
    overrides: dict[str, float],
    num_attrs: int = 40,
    default: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Build a (num_attrs,) condition vector.

    Unspecified attributes default to `default` (0 = attribute absent).
    Use 0.5 for a neutral / ambiguous value.
    """
    c = torch.full((num_attrs,), default, dtype=torch.float32)
    for name, val in overrides.items():
        if name not in ATTR_NAMES:
            raise ValueError(f"Unknown attribute '{name}'. Valid: {ATTR_NAMES}")
        c[ATTR_NAMES.index(name)] = val
    return c.to(device)


def load_image(path: str, img_size: int = 64, device: str = "cuda") -> torch.Tensor:
    """Load and preprocess a single image → (1, 3, H, W) in [-1, 1]."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def save_grid(tensor: torch.Tensor, path: str, nrow: int | None = None):
    """Save a batch of [-1,1] images as a PNG grid."""
    n = tensor.size(0)
    grid = torchvision.utils.make_grid(
        tensor,
        nrow=nrow or n,
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )
    torchvision.utils.save_image(grid, path)
    print(f"Saved → {path}")


# ── commands ──────────────────────────────────────────────────────────────────


def cmd_generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)

    overrides = attrs_from_str(args.attrs) if args.attrs else {}
    c = build_condition(overrides, cfg.num_attrs, default=0.0, device=device)

    imgs = model.generate(c, n=args.n)
    save_grid(imgs, args.out, nrow=args.n)


def cmd_manipulate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)

    x = load_image(args.img, cfg.img_size, device)

    # Source condition: use neutral (0.5) as default — we don't know the
    # true attributes of an arbitrary input image.  Pass --src_attrs to
    # supply known attributes and improve encoding quality.
    src_overrides = attrs_from_str(args.src_attrs) if args.src_attrs else {}
    c_src = build_condition(src_overrides, cfg.num_attrs, default=0.5, device=device)

    c_tgt = c_src.clone()
    for name in args.flip.split(","):
        idx = ATTR_NAMES.index(name.strip())
        c_tgt[idx] = 1.0 - c_tgt[idx]  # binary flip

    original = x
    manipulated = model.manipulate(x, c_src.unsqueeze(0), c_tgt.unsqueeze(0))
    save_grid(torch.cat([original, manipulated]), args.out, nrow=2)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Sample from prior with given attributes")
    gen.add_argument("--ckpt", required=True)
    gen.add_argument("--attrs", default="", help="e.g. 'Smiling=1,Blond_Hair=1'")
    gen.add_argument("--n", type=int, default=8)
    gen.add_argument("--out", default="generated.png")

    man = sub.add_parser("manipulate", help="Flip attributes on a real image")
    man.add_argument("--ckpt", required=True)
    man.add_argument("--img", required=True, help="Path to input face image")
    man.add_argument("--flip", required=True, help="Comma-separated attributes to flip")
    man.add_argument("--src_attrs", default="", help="Known source attributes (optional)")
    man.add_argument("--out", default="manipulated.png")

    args = p.parse_args()
    {"generate": cmd_generate, "manipulate": cmd_manipulate}[args.cmd](args)


if __name__ == "__main__":
    main()
