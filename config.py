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
    beta_max: float = 0.5
    warmup_epochs: int = 25
    grad_clip: float = 1.0
    # Auxiliary attribute-classifier loss to force the decoder to honour c.
    attr_loss_weight: float = 0.5
    # VGG perceptual loss weight (L1 on relu3_3 features); reduces blurriness.
    perceptual_weight: float = 0.1
    # "bf16-mixed" recommended on A100; use "16-mixed" on T4
    precision: str = "bf16-mixed"

    # ── Logging ───────────────────────────────────────────────────────────
    wandb_project: str = "facegen-cvae"
    log_every_n_steps: int = 100
    val_log_n_images: int = 16

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_dir: str = "./checkpoints"
