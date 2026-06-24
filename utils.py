"""Visualisation and dataset analysis helpers."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision

from dataset import ATTR_NAMES


# ── tensor / image utilities ──────────────────────────────────────────────────


def denorm(tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]."""
    return (tensor.clamp(-1, 1) + 1) / 2


def show_grid(
    images: torch.Tensor,
    nrow: int = 8,
    title: str = "",
    figsize: tuple = (14, 4),
):
    """Display a batch of images (CPU or GPU, any normalisation range)."""
    grid = torchvision.utils.make_grid(
        images.cpu(), nrow=nrow, normalize=True, value_range=(-1, 1), padding=2
    )
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ── attribute utilities ───────────────────────────────────────────────────────


def attr_vector(overrides: dict[str, float], default: float = 0.0) -> torch.Tensor:
    """Build a (40,) condition tensor from a dict of name → value overrides."""
    c = torch.full((len(ATTR_NAMES),), default, dtype=torch.float32)
    for name, val in overrides.items():
        c[ATTR_NAMES.index(name)] = float(val)
    return c


def list_attrs() -> None:
    """Print all attribute names with their index."""
    for i, name in enumerate(ATTR_NAMES):
        print(f"  {i:2d}  {name}")


# ── dataset analysis ──────────────────────────────────────────────────────────


def attr_stats(csv_path: str) -> pd.DataFrame:
    """Return a DataFrame with positive-rate and balance score for each attribute."""
    df = pd.read_csv(csv_path)
    rates = ((df[ATTR_NAMES] + 1) / 2).mean()
    balance = 1 - (rates - 0.5).abs() * 2  # 1 = perfectly balanced, 0 = all one value
    return pd.DataFrame({"positive_rate": rates, "balance": balance}).sort_values(
        "balance", ascending=False
    )


def plot_attr_distribution(csv_path: str, top_n: int = 20):
    """Bar chart of positive rates for the top-n most balanced attributes."""
    stats = attr_stats(csv_path).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(stats.index[::-1], stats["positive_rate"][::-1], color="steelblue")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="50%")
    ax.set_xlabel("Positive rate")
    ax.set_title(f"Top-{top_n} most balanced CelebA attributes")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── latent space interpolation ────────────────────────────────────────────────


@torch.no_grad()
def interpolate(
    model,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    c: torch.Tensor,
    steps: int = 8,
) -> torch.Tensor:
    """Linear interpolation between two latent vectors z_a and z_b.

    c: shared condition for all steps (or (steps, num_attrs) for per-step conds).
    Returns (steps, 3, H, W).
    """
    alphas = torch.linspace(0, 1, steps, device=z_a.device)
    z_interp = torch.stack([(1 - a) * z_a + a * z_b for a in alphas])

    if c.dim() == 1:
        c = c.unsqueeze(0).expand(steps, -1)

    return model.decoder(z_interp, c)
