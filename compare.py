"""
Compare v1 (MSE only) vs v2 (FiLM + self-attn + perceptual loss) side by side.

Usage
-----
python compare.py \\
    --ckpt_v1 checkpoints/cvae-epoch099-valloss461.330.ckpt \\
    --ckpt_v2 checkpoints/best.ckpt \\
    --n 8 --seed 0 --out compare.png

Each column is one set of attributes.
Rows (top → bottom): v1 face · v2 face · attribute list.
"""
import argparse

import torch
from PIL import Image, ImageDraw, ImageFont
from config import Config
from dataset import ATTR_NAMES
from generate import random_condition_batch


# ── model loaders ─────────────────────────────────────────────────────────────


def _load_font(size: int):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_v1(ckpt_path: str, device: str):
    """Load old checkpoint into the v1 (frozen) architecture."""
    import model_v1
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = Config(**ckpt["hyper_parameters"])
    m    = model_v1.CVAE(cfg.img_size, cfg.num_attrs, cfg.latent_dim)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    m.load_state_dict(state)
    return m.eval().to(device), cfg


def load_v2(ckpt_path: str, device: str):
    """Load new checkpoint into the v2 (current) architecture."""
    import model as model_v2
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = Config(**ckpt["hyper_parameters"])
    m    = model_v2.CVAE(cfg.img_size, cfg.num_attrs, cfg.latent_dim)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    m.load_state_dict(state)
    return m.eval().to(device), cfg


# ── rendering ─────────────────────────────────────────────────────────────────


def _to_pil(tensor_1chw: torch.Tensor, size: int) -> Image.Image:
    """[-1,1] (1,3,H,W) tensor → upscaled PIL image."""
    arr = ((tensor_1chw.squeeze(0).cpu().float().clamp(-1, 1) + 1) / 2 * 255).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy()).resize((size, size), Image.LANCZOS)


def build_comparison(
    imgs_v1: torch.Tensor,   # (N, 3, H, W)
    imgs_v2: torch.Tensor,
    conditions: torch.Tensor, # (N, 40)  on CPU
    labels: list[str],        # ["v1 — MSE", "v2 — FiLM+Perc"]
    face_size: int = 160,
    font_size: int = 11,
    attr_h: int = 200,
) -> Image.Image:
    n       = imgs_v1.size(0)
    font    = _load_font(font_size)
    lbl_fnt = _load_font(font_size + 2)
    line_h  = font_size + 3
    pad     = 5

    label_w = 130                           # left strip for row labels
    col_w   = face_size
    total_w = label_w + n * col_w
    # rows: label strip | v1 faces | v2 faces | attr text
    row_h   = [face_size, face_size, attr_h]
    total_h = sum(row_h)

    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw   = ImageDraw.Draw(canvas)

    # ── row labels ────────────────────────────────────────────────────────────
    row_colors = [(180, 210, 255), (180, 255, 180), (200, 200, 200)]
    y_offsets  = [0, row_h[0], row_h[0] + row_h[1]]

    for row_idx, (label, color, y0) in enumerate(zip(
        [labels[0], labels[1], "Attributes"], row_colors, y_offsets
    )):
        draw.rectangle([0, y0, label_w - 2, y0 + row_h[row_idx] - 1], fill=(35, 35, 35))
        # vertically centred text
        ty = y0 + row_h[row_idx] // 2 - (font_size + 2) // 2
        draw.text((pad, ty), label, fill=color, font=lbl_fnt)

    # ── column separators ─────────────────────────────────────────────────────
    for col in range(n):
        x0 = label_w + col * col_w
        draw.line([(x0, 0), (x0, total_h - 1)], fill=(60, 60, 60), width=1)

    draw.line([(0, row_h[0]), (total_w, row_h[0])], fill=(80, 80, 80), width=1)
    draw.line([(0, row_h[0] + row_h[1]), (total_w, row_h[0] + row_h[1])], fill=(80, 80, 80), width=1)

    # ── faces ─────────────────────────────────────────────────────────────────
    for col in range(n):
        x0 = label_w + col * col_w

        face_v1 = _to_pil(imgs_v1[col:col+1], face_size)
        canvas.paste(face_v1, (x0, 0))

        face_v2 = _to_pil(imgs_v2[col:col+1], face_size)
        canvas.paste(face_v2, (x0, row_h[0]))

        # ── attributes ────────────────────────────────────────────────────────
        y_attr = row_h[0] + row_h[1]
        draw.rectangle([x0, y_attr, x0 + col_w - 1, total_h - 1], fill=(20, 20, 20))
        active = [
            ATTR_NAMES[j].replace("_", " ")
            for j in range(len(ATTR_NAMES))
            if conditions[col, j].item() >= 0.5
        ]
        ty = y_attr + pad
        for attr in active:
            if ty + line_h > total_h - pad:
                draw.text((x0 + pad, ty), "…", fill=(120, 120, 120), font=font)
                break
            draw.text((x0 + pad, ty), f"• {attr}", fill=(190, 220, 190), font=font)
            ty += line_h

    return canvas


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_v1", required=True, help="Old checkpoint (v1 architecture)")
    p.add_argument("--ckpt_v2", required=True, help="New checkpoint (v2 architecture)")
    p.add_argument("--n",    type=int, default=8,   help="Number of faces to compare")
    p.add_argument("--seed", type=int, default=0,   help="RNG seed for reproducibility")
    p.add_argument("--face_size", type=int, default=160, help="Face tile size in pixels")
    p.add_argument("--out",  default="compare.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading v1…")
    m1, cfg1 = load_v1(args.ckpt_v1, device)
    print("Loading v2…")
    m2, cfg2 = load_v2(args.ckpt_v2, device)

    g = torch.Generator().manual_seed(args.seed)
    # Use the same conditions and the same z for both models → fair comparison
    c = random_condition_batch(args.n, cfg1.num_attrs, device=device, generator=g)

    with torch.no_grad():
        z = torch.randn(args.n, cfg1.latent_dim, device=device)
        imgs_v1 = m1.decoder(z.to(dtype=next(m1.parameters()).dtype), c)
        imgs_v2 = m2.decoder(z.to(dtype=next(m2.parameters()).dtype), c)

    canvas = build_comparison(
        imgs_v1, imgs_v2, c.cpu(),
        labels=["v1 · MSE only", "v2 · FiLM + Perc"],
        face_size=args.face_size,
    )
    canvas.save(args.out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
