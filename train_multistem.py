import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset import RawStems, InfiniteSampler
from models import HTDemucsMultiStem
from losses.gan_loss import GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss
from losses.reconstruction_loss import MultiMelSpecReconstructionLoss

from modules.discriminator.MultiPeriodDiscriminator import MultiPeriodDiscriminator
from modules.discriminator.MultiScaleDiscriminator import MultiScaleDiscriminator
from modules.discriminator.MultiFrequencyDiscriminator import MultiFrequencyDiscriminator
from modules.discriminator.MultiResolutionDiscriminator import MultiResolutionDiscriminator

class CombinedDiscriminator(nn.Module):
    """A wrapper to combine multiple discriminators into a single module."""
    def __init__(self, discriminators_config: List[Dict[str, Any]]):
        super().__init__()
        disc_list = []
        for config in discriminators_config:
            name = config['name']
            params = config['params']
            if name == 'MultiPeriodDiscriminator':
                disc_list.append(MultiPeriodDiscriminator(**params))
            elif name == 'MultiScaleDiscriminator':
                disc_list.append(MultiScaleDiscriminator(**params))
            elif name == 'MultiFrequencyDiscriminator':
                disc_list.append(MultiFrequencyDiscriminator(**params))
            elif name == 'MultiResolutionDiscriminator':
                disc_list.append(MultiResolutionDiscriminator(**params))
            else:
                raise ValueError(f"Unknown discriminator type: {name}")
        self.discriminators = nn.ModuleList(disc_list)

    def forward(self, x: torch.Tensor):
        all_scores, all_fmaps = [], []
        for disc in self.discriminators:
            scores, fmaps = disc(x)
            all_scores.extend(scores)
            all_fmaps.extend(fmaps)
        return all_scores, all_fmaps

class MusicRestorationDataModule(pl.LightningDataModule):
    """Handles data loading for training."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.train_dataset = None

    def setup(self, stage: str | None = None):
        common_params = {
            "sr": self.config['sample_rate'],
            "clip_duration": self.config['clip_duration'],
        }
        self.train_dataset = RawStems(**self.config['train_dataset'], **common_params)

    def train_dataloader(self):
        sampler = InfiniteSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            **self.config['dataloader_params']
        )

class MultiStemMusicRestorationModule(pl.LightningModule):
    """
    PyTorch Lightning module for multi-stem music source restoration.

    This module handles training with generators that output multiple stems
    simultaneously (e.g., HTDemucsMultiStemWrapper).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False

        # 1. Generator (multi-stem output)
        self.generator = self._init_generator()

        # Get number of stems from generator
        self.num_stems = getattr(self.generator, 'num_stems', 1)
        print(f"Generator outputs {self.num_stems} stems")

        # 2. Discriminators (one per stem)
        self.discriminators = nn.ModuleList([
            CombinedDiscriminator(self.hparams.discriminators)
            for _ in range(self.num_stems)
        ])

        # 3. Losses
        loss_cfg = self.hparams.losses
        self.loss_gen_adv = GeneratorLoss(gan_type=loss_cfg.get('gan_type', 'lsgan'))
        self.loss_disc_adv = DiscriminatorLoss(gan_type=loss_cfg.get('gan_type', 'lsgan'))
        self.loss_feat = FeatureMatchingLoss()
        self.loss_recon = MultiMelSpecReconstructionLoss(**loss_cfg['reconstruction_loss'])

    def _init_generator(self):
        model_cfg = self.hparams.model
        if model_cfg['name'] == 'HTDemucsMultiStem':
            return HTDemucsMultiStem.HTDemucsMultiStemWrapper(**model_cfg['params'])
        else:
            raise ValueError(f"Unknown model name: {model_cfg['name']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Get optimizers: 1 for generator, N for discriminators
        optimizers = self.optimizers()
        opt_g = optimizers[0]
        opt_d_list = optimizers[1:]  # One optimizer per discriminator

        target = batch['target']    # (b, c, t)
        mixture = batch['mixture']  # (b, c, t)

        # Reshape from (b, c, t) to (b*c, t) for MSRKit compatibility
        batch_size, num_channels, time_len = target.shape
        target = rearrange(target, 'b c t -> (b c) t')
        mixture = rearrange(mixture, 'b c t -> (b c) t')

        # Generate: (b*c, t) -> (b*c, N, t) where N = num_stems
        generated = self(mixture)

        # --- Train Discriminators (one per stem) ---
        d_losses = []
        for stem_idx in range(self.num_stems):
            # Extract stem: (b*c, N, t) -> (b*c, t)
            generated_stem = generated[:, stem_idx, :]

            # Discriminator expects (b*c, 1, t)
            real_scores, _ = self.discriminators[stem_idx](target.unsqueeze(1))
            fake_scores, _ = self.discriminators[stem_idx](generated_stem.detach().unsqueeze(1))

            d_loss, _, _ = self.loss_disc_adv(real_scores, fake_scores)

            opt_d_list[stem_idx].zero_grad()
            self.manual_backward(d_loss)
            opt_d_list[stem_idx].step()

            d_losses.append(d_loss)
            self.log(f'train/d_loss_stem{stem_idx}', d_loss)

        # Average discriminator loss
        d_loss_avg = torch.stack(d_losses).mean()
        self.log('train/d_loss', d_loss_avg, prog_bar=True)

        # --- Train Generator ---
        g_losses_adv = []
        g_losses_feat = []
        g_losses_recon = []

        for stem_idx in range(self.num_stems):
            generated_stem = generated[:, stem_idx, :]

            # Adversarial and feature matching
            real_scores, real_fmaps = self.discriminators[stem_idx](target.unsqueeze(1))
            fake_scores, fake_fmaps = self.discriminators[stem_idx](generated_stem.unsqueeze(1))

            loss_adv, _ = self.loss_gen_adv(fake_scores)
            loss_feat = self.loss_feat(real_fmaps, fake_fmaps)

            # Reconstruction loss
            loss_recon = self.loss_recon(generated_stem, target)

            g_losses_adv.append(loss_adv)
            g_losses_feat.append(loss_feat)
            g_losses_recon.append(loss_recon)

            self.log(f'train/g_adv_stem{stem_idx}', loss_adv)
            self.log(f'train/g_feat_stem{stem_idx}', loss_feat)
            self.log(f'train/g_recon_stem{stem_idx}', loss_recon)

        # Average losses across stems
        loss_cfg = self.hparams.losses
        g_loss = (
            torch.stack(g_losses_recon).mean() * loss_cfg['lambda_recon'] +
            torch.stack(g_losses_adv).mean() * loss_cfg['lambda_gan'] +
            torch.stack(g_losses_feat).mean() * loss_cfg['lambda_feat']
        )

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log('train/g_loss', g_loss, prog_bar=True)
        self.log('train/g_recon', torch.stack(g_losses_recon).mean(), prog_bar=True)
        self.log('train/g_adv', torch.stack(g_losses_adv).mean())
        self.log('train/g_feat', torch.stack(g_losses_feat).mean())

    def configure_optimizers(self):
        opt_g_cfg = self.hparams.optimizer_g
        opt_d_cfg = self.hparams.optimizer_d

        # Generator optimizer
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                lr=opt_g_cfg['lr'], betas=opt_g_cfg['betas'])

        # One optimizer per discriminator
        opt_d_list = [
            torch.optim.Adam(disc.parameters(),
                           lr=opt_d_cfg['lr'], betas=opt_d_cfg['betas'])
            for disc in self.discriminators
        ]

        return [opt_g] + opt_d_list

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Data module
    data_module = MusicRestorationDataModule(config['data'])

    # Model
    model = MultiStemMusicRestorationModule(config)

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=f"{config['project_name']}/{config['exp_name']}"
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config['project_name']}/{config['exp_name']}",
        filename='{epoch}-{step}',
        every_n_train_steps=config['trainer']['checkpoint_save_interval'],
        save_top_k=-1,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        max_steps=config['trainer']['max_steps'],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    )

    # Train
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    main(args)
