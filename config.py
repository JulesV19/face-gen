from dataclasses import dataclass


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "./data"
    img_size: int = 64
    num_attrs: int = 40
    num_workers: int = 4

    # ── Model ─────────────────────────────────────────────────────────────
    # Encoder: 3→64→128→256→512 (stride-2 conv × 4); decoder mirrors it.
    latent_dim: int = 256

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    # β-VAE weight on KL term; linear warmup from 0 → beta_max over warmup_epochs
    beta_max: float = 1.0
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    # Auxiliary attribute-classifier loss to force the decoder to honour c.
    # 0 = disabled (default, unchanged behaviour). Try 0.1–1.0 to enable.
    attr_loss_weight: float = 0.0
    # "bf16-mixed" recommended on A100; use "16-mixed" on T4
    precision: str = "bf16-mixed"

    # ── Logging ───────────────────────────────────────────────────────────
    wandb_project: str = "facegen-cvae"
    log_every_n_steps: int = 100
    val_log_n_images: int = 16

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_dir: str = "./checkpoints"


@dataclass
class DiffusionConfig:
    """Conditional DDPM on CelebA — produces sharp faces (unlike the VAE)."""

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "./data"
    img_size: int = 64
    num_attrs: int = 40
    num_workers: int = 4

    # ── Model (UNet) ──────────────────────────────────────────────────────
    base_channels: int = 128
    channel_mults: tuple = (1, 2, 2, 2)  # resolutions 64 → 32 → 16 → 8
    num_res_blocks: int = 2
    attn_resolutions: tuple = (16, 8)
    dropout: float = 0.1

    # ── Diffusion ─────────────────────────────────────────────────────────
    timesteps: int = 1000
    schedule: str = "cosine"  # "cosine" | "linear"
    # Classifier-free guidance: prob. of dropping the condition during training
    cond_drop_prob: float = 0.1

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 100
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    precision: str = "bf16-mixed"

    # ── Sampling / logging ────────────────────────────────────────────────
    wandb_project: str = "facegen-diffusion"
    log_every_n_steps: int = 100
    sample_every_n_epochs: int = 5
    n_sample_images: int = 16
    guidance_scale: float = 3.0  # CFG strength at sampling (1 = no guidance)
    ddim_steps: int = 50

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_dir: str = "./checkpoints_diffusion"
