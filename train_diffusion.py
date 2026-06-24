"""
Training entry-point for the conditional diffusion model.

    python train_diffusion.py --data_dir /content/data --max_epochs 100

Hyperparameters are saved as flat hparams so that generate_diffusion.py can
rebuild the model from a checkpoint without importing this training stack.
"""
import argparse
import copy
import dataclasses

import torch
import torchvision
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from config import DiffusionConfig
from dataset import CelebADataset
from diffusion import GaussianDiffusion
from unet import UNet


# ── Lightning module ──────────────────────────────────────────────────────────


class DiffusionModule(pl.LightningModule):
    def __init__(self, **cfg_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg = DiffusionConfig(**cfg_kwargs)

        self.unet = UNet(
            img_size=cfg.img_size,
            num_attrs=cfg.num_attrs,
            base_channels=cfg.base_channels,
            channel_mults=tuple(cfg.channel_mults),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=tuple(cfg.attn_resolutions),
            dropout=cfg.dropout,
        )
        # EMA copy — sampled from for stable, higher-quality images. Kept in the
        # checkpoint; generate_diffusion.py loads these weights, not the raw ones.
        self.ema_unet = copy.deepcopy(self.unet)
        self.ema_unet.requires_grad_(False)

        self.diffusion = GaussianDiffusion(cfg.timesteps, cfg.schedule)
        self._val_attrs: torch.Tensor | None = None

    # ── training ──────────────────────────────────────────────────────────

    def training_step(self, batch, _):
        x, c = batch
        loss = self.diffusion.p_losses(self.unet, x, c, self.cfg.cond_drop_prob)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def on_train_batch_end(self, *_):
        d = self.cfg.ema_decay
        for ema_p, p in zip(self.ema_unet.parameters(), self.unet.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for ema_b, b in zip(self.ema_unet.buffers(), self.unet.buffers()):
            ema_b.copy_(b)

    # ── validation ────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        x, c = batch
        if batch_idx == 0:
            self._val_attrs = c[: self.cfg.n_sample_images].clone()
        loss = self.diffusion.p_losses(self.unet, x, c, cond_drop_prob=0.0)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if (
            self.trainer.sanity_checking
            or self._val_attrs is None
            or not isinstance(self.logger, WandbLogger)
            or (self.current_epoch + 1) % self.cfg.sample_every_n_epochs != 0
        ):
            return

        attrs = self._val_attrs.to(self.device)
        samples = self.diffusion.ddim_sample(
            self.ema_unet,
            attrs,
            guidance_scale=self.cfg.guidance_scale,
            ddim_steps=self.cfg.ddim_steps,
            img_size=self.cfg.img_size,
        )
        grid = torchvision.utils.make_grid(
            samples, nrow=8, normalize=True, value_range=(-1, 1)
        )
        self.logger.experiment.log(
            {"val/samples": wandb.Image(grid), "epoch": self.current_epoch}
        )

    # ── optimiser ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.unet.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ── data ──────────────────────────────────────────────────────────────

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        ds = CelebADataset(self.cfg.data_dir, split, self.cfg.img_size)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse() -> DiffusionConfig:
    defaults = DiffusionConfig()
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(defaults):
        if isinstance(f.default, (tuple, list)):
            continue  # structural fields (channel_mults, …) stay at defaults
        p.add_argument(f"--{f.name}", type=type(f.default), default=None)
    args = p.parse_args()

    cfg = DiffusionConfig()
    for f in dataclasses.fields(cfg):
        if isinstance(f.default, (tuple, list)):
            continue
        v = getattr(args, f.name)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


def main():
    cfg = _parse()

    wandb_logger = WandbLogger(project=cfg.wandb_project, log_model=False)

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.ckpt_dir,
            filename="diffusion-epoch{epoch:03d}-valloss{val/loss:.4f}",
            monitor="val/loss",
            save_top_k=3,
            mode="min",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=cfg.log_every_n_steps),
    ]

    module = DiffusionModule(**dataclasses.asdict(cfg))

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg.log_every_n_steps,
    )
    trainer.fit(module)


if __name__ == "__main__":
    main()
