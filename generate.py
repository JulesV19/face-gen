"""
Inference utilities — generation and attribute manipulation.

Examples
--------
# generate 8 blond smiling young faces
python generate.py generate \\
    --ckpt checkpoints/best.ckpt \\
    --attrs "Blond_Hair=1,Smiling=1,Young=1" \\
    --n 8 --out generated.png

# generate a big gallery of varied, randomly-characterised faces
python generate.py gallery \\
    --ckpt checkpoints/best.ckpt \\
    --n 64 --cols 8 --out gallery.png

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
from PIL import Image, ImageDraw, ImageFont
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


def random_condition_batch(
    n: int,
    num_attrs: int = 40,
    device: str = "cuda",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build (n, num_attrs) condition vectors with diverse *coherent* faces.

    Independent Bernoulli over 40 attributes yields incoherent monsters
    (blond AND black hair, male with heavy makeup, …).  Instead we sample
    structured, mutually-consistent attribute sets per face:

      • exactly one hair colour (or none → bald)
      • gender, with correlated grooming (beard/makeup/lipstick)
      • age, expression, and a few low-probability accessories.
    """
    def idx(name: str) -> int:
        return ATTR_NAMES.index(name)

    g = generator
    c = torch.zeros(n, num_attrs, device="cpu")

    for i in range(n):
        male = torch.rand(1, generator=g).item() < 0.5
        young = torch.rand(1, generator=g).item() < 0.7

        c[i, idx("Male")] = float(male)
        c[i, idx("Young")] = float(young)

        # ── hair: pick one colour bucket (with a chance of bald) ──────────
        hair_opts = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]
        roll = torch.rand(1, generator=g).item()
        if male and roll < 0.05:
            c[i, idx("Bald")] = 1.0
        else:
            pick = int(torch.randint(len(hair_opts), (1,), generator=g).item())
            c[i, idx(hair_opts[pick])] = 1.0
        if torch.rand(1, generator=g).item() < 0.4:
            c[i, idx("Wavy_Hair" if torch.rand(1, generator=g).item() < 0.5
                     else "Straight_Hair")] = 1.0

        # ── grooming, correlated with gender ──────────────────────────────
        if male:
            if torch.rand(1, generator=g).item() < 0.4:
                c[i, idx("No_Beard")] = 0.0
                c[i, idx("Mustache")] = float(torch.rand(1, generator=g).item() < 0.4)
                c[i, idx("Goatee")] = float(torch.rand(1, generator=g).item() < 0.4)
                c[i, idx("5_o_Clock_Shadow")] = float(torch.rand(1, generator=g).item() < 0.5)
            else:
                c[i, idx("No_Beard")] = 1.0
        else:
            c[i, idx("No_Beard")] = 1.0
            heavy = torch.rand(1, generator=g).item() < 0.6
            c[i, idx("Heavy_Makeup")] = float(heavy)
            c[i, idx("Wearing_Lipstick")] = float(heavy or torch.rand(1, generator=g).item() < 0.4)
            c[i, idx("Wearing_Earrings")] = float(torch.rand(1, generator=g).item() < 0.3)
            c[i, idx("Arched_Eyebrows")] = float(torch.rand(1, generator=g).item() < 0.4)

        # ── expression ────────────────────────────────────────────────────
        smiling = torch.rand(1, generator=g).item() < 0.5
        c[i, idx("Smiling")] = float(smiling)
        c[i, idx("High_Cheekbones")] = float(smiling)
        c[i, idx("Mouth_Slightly_Open")] = float(smiling and torch.rand(1, generator=g).item() < 0.6)

        # ── occasional accessories ────────────────────────────────────────
        c[i, idx("Eyeglasses")] = float(torch.rand(1, generator=g).item() < 0.2)
        c[i, idx("Wearing_Hat")] = float(torch.rand(1, generator=g).item() < 0.1)
        if not young:
            c[i, idx("Gray_Hair")] = float(torch.rand(1, generator=g).item() < 0.4)

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


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def save_annotated_gallery(
    imgs: torch.Tensor,
    conditions: torch.Tensor,
    path: str,
    ncols: int = 4,
    face_size: int = 128,
    text_w: int = 220,
    font_size: int = 9,
):
    """Save a gallery where each face cell shows the face + its active attributes."""
    font = _load_font(font_size)
    line_h = font_size + 3
    padding = 5
    cell_w = face_size + text_w
    cell_h = face_size

    n = imgs.size(0)
    nrows = (n + ncols - 1) // ncols

    canvas = Image.new("RGB", (cell_w * ncols, cell_h * nrows), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # [-1,1] → uint8
    imgs_u8 = ((imgs.cpu().float().clamp(-1, 1) + 1) / 2 * 255).byte()

    for i in range(n):
        row, col = divmod(i, ncols)
        x0, y0 = col * cell_w, row * cell_h

        # ── face ──────────────────────────────────────────────────────────
        face = Image.fromarray(imgs_u8[i].permute(1, 2, 0).numpy()).resize(
            (face_size, face_size), Image.LANCZOS
        )
        canvas.paste(face, (x0, y0))

        # ── attribute list ─────────────────────────────────────────────────
        tx0 = x0 + face_size
        draw.rectangle([tx0, y0, tx0 + text_w - 1, y0 + cell_h - 1], fill=(20, 20, 20))

        active = [
            ATTR_NAMES[j].replace("_", " ")
            for j in range(len(ATTR_NAMES))
            if conditions[i, j].item() >= 0.5
        ]

        ty = y0 + padding
        for attr in active:
            if ty + line_h > y0 + cell_h - padding:
                break
            draw.text((tx0 + padding, ty), f"• {attr}", fill=(200, 230, 200), font=font)
            ty += line_h

        # subtle separator
        draw.line([(x0, y0), (x0, y0 + cell_h - 1)], fill=(60, 60, 60), width=1)
        draw.line([(x0, y0), (x0 + cell_w - 1, y0)], fill=(60, 60, 60), width=1)

    canvas.save(path)
    print(f"Saved → {path}")


# ── commands ──────────────────────────────────────────────────────────────────


def cmd_generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)

    overrides = attrs_from_str(args.attrs) if args.attrs else {}
    c = build_condition(overrides, cfg.num_attrs, default=0.0, device=device)

    imgs = model.generate(c, n=args.n)
    save_grid(imgs, args.out, nrow=args.n)


def cmd_gallery(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)

    g = None
    if args.seed is not None:
        g = torch.Generator().manual_seed(args.seed)

    c = random_condition_batch(args.n, cfg.num_attrs, device=device, generator=g)
    imgs = model.generate(c)
    save_annotated_gallery(
        imgs, c.cpu(), args.out, ncols=args.cols, face_size=args.face_size,
        text_w=240, font_size=12,
    )


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

    gal = sub.add_parser(
        "gallery", help="Sample many faces, each with different random attributes"
    )
    gal.add_argument("--ckpt", required=True)
    gal.add_argument("--n", type=int, default=32, help="Number of faces")
    gal.add_argument("--cols", type=int, default=4, help="Faces per row in the grid")
    gal.add_argument("--face-size", type=int, default=128, help="Face tile size in px")
    gal.add_argument("--seed", type=int, default=None, help="Reproducible sampling")
    gal.add_argument("--out", default="gallery.png")

    man = sub.add_parser("manipulate", help="Flip attributes on a real image")
    man.add_argument("--ckpt", required=True)
    man.add_argument("--img", required=True, help="Path to input face image")
    man.add_argument("--flip", required=True, help="Comma-separated attributes to flip")
    man.add_argument("--src_attrs", default="", help="Known source attributes (optional)")
    man.add_argument("--out", default="manipulated.png")

    args = p.parse_args()
    {
        "generate": cmd_generate,
        "gallery": cmd_gallery,
        "manipulate": cmd_manipulate,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
