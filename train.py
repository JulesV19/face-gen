"""
Training entry-point.

    python train.py --data_dir /content/data --max_epochs 100

All Config fields are saved as flat hparams so that
    CVAEModule.load_from_checkpoint(path)
works without any extra hooks in generate.py.
"""
import argparse
import dataclasses

import torch
import torch.nn.functional as F
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

from config import Config
from dataset import CelebADataset
from model import AttrClassifier, CVAE


# ── Perceptual loss ───────────────────────────────────────────────────────────


class VGGPerceptual(torch.nn.Module):
    """L1 loss on relu3_3 features of a frozen VGG-16.

    Inputs are expected in [-1, 1]; they are renormalised to ImageNet stats
    before being passed through the network.
    """

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        self.features = torch.nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        def prep(t: torch.Tensor) -> torch.Tensor:
            t = (t.clamp(-1, 1) + 1) / 2          # [-1,1] → [0,1]
            return (t - self.mean) / self.std       # ImageNet normalisation
        return F.l1_loss(self.features(prep(x)), self.features(prep(y)))


# ── Lightning module ──────────────────────────────────────────────────────────


class CVAEModule(pl.LightningModule):
    def __init__(self, **cfg_kwargs):
        super().__init__()
        # Store as flat primitives so PL can serialize/deserialize cleanly
        self.save_hyperparameters()
        self.cfg = Config(**cfg_kwargs)
        self.model = CVAE(
            img_size=self.cfg.img_size,
            num_attrs=self.cfg.num_attrs,
            latent_dim=self.cfg.latent_dim,
        )
        # Optional — built only when the auxiliary attribute loss is enabled.
        self.aux = (
            AttrClassifier(self.cfg.img_size, self.cfg.num_attrs)
            if self.cfg.attr_loss_weight > 0
            else None
        )
        self.perceptual = VGGPerceptual()
        self._val_fixed: tuple | None = None

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x, c):
        return self.model(x, c)

    # ── KL schedule ───────────────────────────────────────────────────────

    def _beta(self) -> float:
        progress = min(1.0, self.current_epoch / max(1, self.cfg.warmup_epochs))
        return self.cfg.beta_max * progress

    # ── shared step ───────────────────────────────────────────────────────

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, c = batch
        recon, mu, logvar = self(x, c)

        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        perc_loss  = self.perceptual(recon, x)
        kl_loss = (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        )
        loss = recon_loss + self.cfg.perceptual_weight * perc_loss + self._beta() * kl_loss

        logs = {
            f"{stage}/loss":  loss,
            f"{stage}/recon": recon_loss,
            f"{stage}/perc":  perc_loss,
            f"{stage}/kl":    kl_loss,
            f"{stage}/beta":  self._beta(),
        }

        if self.aux is not None:
            # cls_loss trains the classifier on real images; attr_loss pushes the
            # decoder so the classifier reads c back off the reconstruction.
            cls_loss = F.binary_cross_entropy_with_logits(self.aux(x), c)
            attr_loss = F.binary_cross_entropy_with_logits(self.aux(recon), c)
            loss = loss + cls_loss + self.cfg.attr_loss_weight * attr_loss
            logs[f"{stage}/loss"] = loss
            logs[f"{stage}/cls"] = cls_loss
            logs[f"{stage}/attr"] = attr_loss

        on_step = stage == "train"
        self.log_dict(
            logs,
            prog_bar=on_step,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self._val_fixed = batch
        return self._step(batch, "val")

    # ── image logging ─────────────────────────────────────────────────────

    def on_validation_epoch_end(self):
        if self._val_fixed is None or not isinstance(self.logger, WandbLogger):
            return

        x, c = self._val_fixed
        n = min(self.cfg.val_log_n_images, x.size(0))
        x, c = x[:n].to(self.device), c[:n].to(self.device)

        with torch.no_grad():
            recon, _, _ = self.model(x, c)

        grid = torchvision.utils.make_grid(
            torch.cat([x, recon]),
            nrow=n,
            normalize=True,
            value_range=(-1, 1),
        )
        self.logger.experiment.log(
            {"val/reconstructions": wandb.Image(grid), "epoch": self.current_epoch}
        )

    # ── optimiser ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.max_epochs, eta_min=1e-5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ── data ──────────────────────────────────────────────────────────────

    def train_dataloader(self):
        ds = CelebADataset(self.cfg.data_dir, "train", self.cfg.img_size)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self):
        ds = CelebADataset(self.cfg.data_dir, "val", self.cfg.img_size)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse() -> Config:
    defaults = Config()
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(defaults):
        p.add_argument(f"--{f.name}", type=type(f.default), default=None)
    args = p.parse_args()

    cfg = Config()
    for f in dataclasses.fields(cfg):
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
            filename="cvae-epoch{epoch:03d}-valloss{val/loss:.3f}",
            monitor="val/loss",
            save_top_k=3,
            mode="min",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        # In Colab, train.py runs via `!python …` so stdout is not a TTY: a
        # refresh_rate of 1 makes tqdm print a fresh line every step (thousands
        # of lines). Throttle it — real metrics are in wandb anyway.
        TQDMProgressBar(refresh_rate=cfg.log_every_n_steps),
    ]

    module = CVAEModule(**dataclasses.asdict(cfg))

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
