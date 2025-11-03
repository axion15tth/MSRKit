import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# ---- Datasets ----
from data.dataset import RawStems, InfiniteSampler  # 既存
# filelist 方式があれば使う（無ければ RawStems を使用）
try:
    from data.datasets import FileListDataset, FileListInfiniteSampler
    HAS_FILELIST = True
except Exception:
    FileListDataset, FileListInfiniteSampler = None, None
    HAS_FILELIST = False

# ---- Generators ----
from models import MelRNN, MelRoFormer, UNet, HTDemucs  # 既存
# 新: HTDemucs → Complex U-Net 統合ジェネレーター
from models.htdemucs_cunet import HTDemucsCUNetGenerator

# ---- Adversarial losses (既存) ----
from losses.gan_loss import GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss

# ---- Objective / Perceptual losses (新/既存) ----
from losses.sisnr import si_snr_loss  # 追加済み想定: (pred, target) -> scalar（最小化用に負 SI-SNR を返す or 1/SSNR）
from losses.mrstft import MultiResolutionSTFTLoss  # 追加済み想定: (pred, target) -> scalar
from losses.fad_clap_approx import (
    CLAPEmbedLoss, CLAPEmbedLossConfig,
    FADProxyLoss, FADProxyConfig,
)
from losses.consistency import MixtureReconstructLoss

# ---- Discriminators ----
from modules.discriminator.MultiPeriodDiscriminator import MultiPeriodDiscriminator
from modules.discriminator.MultiScaleDiscriminator import MultiScaleDiscriminator
from modules.discriminator.MultiFrequencyDiscriminator import MultiFrequencyDiscriminator
from modules.discriminator.MultiResolutionDiscriminator import MultiResolutionDiscriminator


# ============================================================
# Utilities
# ============================================================

def _get(d: Dict[str, Any], path: str, default=None):
    """Nested getter: 'a.b.c' -> value"""
    cur = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _to_mono(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) -> (B, 1, T) mono"""
    return x.mean(dim=1, keepdim=True)

def _disc_input(x: torch.Tensor) -> torch.Tensor:
    """Prepare input for discriminators: expects (B, 1, T)."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.size(1) != 1:
        return _to_mono(x)
    return x


# ============================================================
# Combined Discriminator
# ============================================================

class CombinedDiscriminator(nn.Module):
    """Combine multiple discriminators into a single module."""
    def __init__(self, discriminators_config: List[Dict[str, Any]]):
        super().__init__()
        disc_list = []
        for config in discriminators_config:
            name = config['name']
            params = config.get('params', {})
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
        # require (B,1,T)
        x = _disc_input(x)
        all_scores, all_fmaps = [], []
        for disc in self.discriminators:
            scores, fmaps = disc(x)
            all_scores.extend(scores)
            all_fmaps.extend(fmaps)
        return all_scores, all_fmaps


# ============================================================
# DataModule
# ============================================================

class MusicRestorationDataModule(pl.LightningDataModule):
    """Handles data loading for training."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.train_dataset = None

    def setup(self, stage: str | None = None):
        sr = self.config.get('sample_rate', 48_000)
        clip_duration = float(self.config.get('clip_duration', 10.0))

        # filelist 指定があれば優先
        if HAS_FILELIST and ('train_file_list' in self.config):
            self.train_dataset = FileListDataset(
                filelist=self.config['train_file_list'],
                sr=sr,
                clip_duration=clip_duration,
            )
            sampler = FileListInfiniteSampler(self.train_dataset)
            self._sampler = sampler
        else:
            # fallback: 既存 RawStems
            common_params = {"sr": sr, "clip_duration": clip_duration}
            self.train_dataset = RawStems(**self.config['train_dataset'], **common_params)
            sampler = InfiniteSampler(self.train_dataset)
            self._sampler = sampler
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self._sampler,
            **self.config.get('dataloader_params', {"batch_size": 8, "num_workers": 8})
        )


# ============================================================
# LightningModule
# ============================================================

class MusicRestorationModule(pl.LightningModule):
    """
    PyTorch Lightning module for music source restoration.
    - Generator: HTDemucs → Complex U-Net（推奨）/ 既存モデル互換
    - Losses: SI-SNR / L1 / MR-STFT / (CLAP, FAD-proxy, Mixture-Reconstruct) + GAN/FM
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # そのまま保存（Logger 用）
        self.save_hyperparameters(config)
        self.automatic_optimization = False  # GAN 手動最適化

        # ---- Generator
        self.generator = self._init_generator()

        # ---- Discriminator
        self.discriminator = CombinedDiscriminator(self.hparams.get('discriminators', []))

        # ---- Loss configs（後方互換: 'loss' が無ければ 'losses' を読む）
        loss_cfg = self.hparams.get('loss', self.hparams.get('losses', {}))
        self.loss_cfg = loss_cfg

        # ---- Objective losses
        # SI-SNR（関数）
        self.use_sisnr = float(loss_cfg.get('sisnr', 0.0)) > 0
        # L1（時間波形）
        self.use_l1 = float(loss_cfg.get('l1_wave', 0.0)) > 0
        # MR-STFT
        mr = loss_cfg.get('mrstft', None)
        if isinstance(mr, dict):
            fft_sizes = mr.get('windows', [4096, 1024, 256])
            hop_sizes = mr.get('hops', [1024, 256, 64])
            self.mrstft = MultiResolutionSTFTLoss(
                fft_sizes=fft_sizes,
                hop_sizes=hop_sizes,
                win_lengths=fft_sizes,  # win_length = fft_size (default)
            )
            self.w_mrstft = float(mr.get('weight', loss_cfg.get('mrstft', 0.0) or 0.0))
        else:
            self.mrstft = None
            self.w_mrstft = float(loss_cfg.get('mrstft', 0.0) or 0.0)

        # ---- Perceptual: CLAP / FAD proxy
        self.w_clap = float(loss_cfg.get('clap_embed', 0.0) or 0.0)
        self.w_fad  = float(loss_cfg.get('fad_proxy', 0.0) or 0.0)
        if self.w_clap > 0:
            self.clap_loss = CLAPEmbedLoss(CLAPEmbedLossConfig(
                sample_rate=self.hparams.get('sample_rate', 48_000),
                distance="cosine"
            ))
        if self.w_fad > 0:
            self.fad_loss = FADProxyLoss(FADProxyConfig(
                sample_rate=self.hparams.get('sample_rate', 48_000),
                ref_stats_path=loss_cfg.get('fad_ref_stats', None),
                diag_only=True
            ))

        # ---- Consistency: Mixture reconstruct
        self.w_mixrec = float(loss_cfg.get('mixture_reconstruct', 0.0) or 0.0)
        if self.w_mixrec > 0:
            self.mixrec_loss = MixtureReconstructLoss(reduction="mean")
        # degradations（任意）
        self.forward_degrade = None
        try:
            from data.degradations import forward_degrade  # 微分可能近似チェーン
            self.forward_degrade = forward_degrade
        except Exception:
            pass

        # ---- Adversarial losses（既存）
        # NOTE: 'losses' 互換のためキー名に配慮
        gan_type = loss_cfg.get('gan', {}).get('type', loss_cfg.get('gan_type', 'hinge'))
        self.loss_gen_adv = GeneratorLoss(gan_type=gan_type)
        self.loss_disc_adv = DiscriminatorLoss(gan_type=gan_type)
        self.loss_feat = FeatureMatchingLoss()
        self.w_gan = float(loss_cfg.get('gan', {}).get('weight', loss_cfg.get('lambda_gan', 0.0)))
        self.w_fm  = float(loss_cfg.get('gan', {}).get('fm_weight', loss_cfg.get('lambda_feat', 0.0)))

        # L1 weight（互換）
        self.w_l1 = float(loss_cfg.get('l1_wave', loss_cfg.get('lambda_recon', 0.0)))
        # SI-SNR weight
        self.w_sisnr = float(loss_cfg.get('sisnr', 0.0))

    # -------------------------
    # Model selection
    # -------------------------
    def _init_generator(self):
        model_cfg = self.hparams['model']
        name = model_cfg['name']
        params = model_cfg.get('params', {})
        if name == 'MelRNN':
            return MelRNN.MelRNN(**params)
        elif name == 'MelRoFormer':
            return MelRoFormer.MelRoFormer(**params)
        elif name == 'MelUNet':
            return UNet.MelUNet(**params)
        elif name == 'HTDemucs':
            return HTDemucs.HTDemucsWrapper(**params)
        elif name == 'HTDemucsCUNetGenerator':
            # 新モデル（推奨）
            return HTDemucsCUNetGenerator(self.hparams)
        else:
            raise ValueError(f"Unknown model name: {name}")

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, T) mixture
        return: (B, 2, T) restored target stem
        """
        return self.generator(x)

    # -------------------------
    # Training
    # -------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        opt_g, opt_d = self.optimizers()

        target: torch.Tensor = batch['target']          # (B, 2, T) clean stem
        mixture_clean: torch.Tensor = batch['mixture']  # (B, 2, T) clean mix

        # ---- Apply degradation (on-the-fly) ----
        mixture_in = mixture_clean
        degrade_preset = None
        if self.forward_degrade is not None:
            mixture_in, degrade_preset = self.forward_degrade(
                mixture_clean, mode="random", return_params=True
            )

        # ---- Discriminator step ----
        with torch.cuda.amp.autocast(enabled=False):
            generated: torch.Tensor = self(mixture_in)  # (B, 2, T)

        real_scores, _ = self.discriminator(target)             # expects (B,1,T) internally
        fake_scores, _ = self.discriminator(generated.detach())

        d_loss, _, _ = self.loss_disc_adv(real_scores, fake_scores)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()
        self.log('train/d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=False)

        # ---- Generator step ----
        real_scores, real_fmaps = self.discriminator(target)
        fake_scores, fake_fmaps = self.discriminator(generated)

        # Objective losses
        g_loss_total = 0.0

        # L1 waveform
        if self.w_l1 > 0:
            loss_l1 = F.l1_loss(generated, target)
            g_loss_total = g_loss_total + self.w_l1 * loss_l1
            self.log('train/l1', loss_l1, on_step=True, on_epoch=False)

        # SI-SNR
        if self.w_sisnr > 0:
            loss_sisnr = si_snr_loss(generated, target)
            g_loss_total = g_loss_total + self.w_sisnr * loss_sisnr
            self.log('train/sisnr', loss_sisnr, on_step=True, on_epoch=False)

        # MR-STFT
        if self.mrstft is not None and self.w_mrstft > 0:
            loss_mrstft = self.mrstft(generated, target)
            g_loss_total = g_loss_total + self.w_mrstft * loss_mrstft
            self.log('train/mrstft', loss_mrstft, on_step=True, on_epoch=False)

        # Perceptual: CLAP embedding distance
        if self.w_clap > 0:
            with torch.no_grad():  # CLAP は勾配不要・fp32で
                loss_clap = self.clap_loss(generated, target)
            g_loss_total = g_loss_total + self.w_clap * loss_clap
            self.log('train/clap', loss_clap, on_step=True, on_epoch=False)

        # Perceptual: FAD proxy
        if self.w_fad > 0:
            with torch.no_grad():
                # 参照統計があれば pred のみ、無ければ GT と比較
                if hasattr(self.fad_loss, "mu_ref") and self.fad_loss.mu_ref.numel() > 0:
                    loss_fad = self.fad_loss(generated)
                else:
                    loss_fad = self.fad_loss(generated, ref_wave=target)
            g_loss_total = g_loss_total + self.w_fad * loss_fad
            self.log('train/fad_proxy', loss_fad, on_step=True, on_epoch=False)

        # Consistency: Mixture reconstruct
        if self.w_mixrec > 0:
            # Use the same degrade preset to match the input degradation
            degrade_kwargs = {"preset": degrade_preset} if degrade_preset is not None else None
            loss_mixrec = self.mixrec_loss(
                mix=mixture_in,  # Use degraded mixture
                pred_stem=generated,
                gt_stem=target,
                degrade_fn=self.forward_degrade if self.forward_degrade is not None else None,
                degrade_kwargs=degrade_kwargs,
                apply_degrade_to_mix=False
            )
            g_loss_total = g_loss_total + self.w_mixrec * loss_mixrec
            self.log('train/mixrec', loss_mixrec, on_step=True, on_epoch=False)

        # Adversarial + Feature Matching
        loss_adv, _ = self.loss_gen_adv(fake_scores)
        loss_fm = self.loss_feat(real_fmaps, fake_fmaps)

        g_loss_total = g_loss_total + self.w_gan * loss_adv + self.w_fm * loss_fm

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss_total)
        opt_g.step()

        self.log('train/g_loss', g_loss_total, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train/adv', loss_adv, on_step=True, on_epoch=False)
        self.log('train/fm', loss_fm, on_step=True, on_epoch=False)

        # Step schedulers（ウォームアップ等）
        scheds = self.lr_schedulers()
        if isinstance(scheds, (list, tuple)) and len(scheds) == 2:
            sch_g, sch_d = scheds
            if sch_g: sch_g.step()
            if sch_d: sch_d.step()

    # -------------------------
    # Optimizers / Schedulers
    # -------------------------
    def configure_optimizers(self):
        # Generator Optimizer
        opt_g_cfg = self.hparams.get('optimizer_g', {"lr": 2e-4, "betas": [0.9, 0.95]})
        opt_g = torch.optim.AdamW(self.generator.parameters(),
                                  lr=opt_g_cfg['lr'],
                                  betas=tuple(opt_g_cfg.get('betas', (0.9, 0.95))),
                                  weight_decay=float(opt_g_cfg.get('weight_decay', 0.0)))
        
        # Discriminator Optimizer
        opt_d_cfg = self.hparams.get('optimizer_d', {"lr": 2e-4, "betas": [0.9, 0.95]})
        opt_d = torch.optim.AdamW(self.discriminator.parameters(),
                                  lr=opt_d_cfg['lr'],
                                  betas=tuple(opt_d_cfg.get('betas', (0.9, 0.95))),
                                  weight_decay=float(opt_d_cfg.get('weight_decay', 0.0)))

        # Warmup scheduler（オプション）
        sch_cfg = self.hparams.get('scheduler', {})
        if 'warm_up_steps' in sch_cfg:
            warmup_steps = int(sch_cfg['warm_up_steps'])
            lr_lambda = lambda step: min(1.0, (step + 1) / max(1, warmup_steps))
            scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
            scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)
            return [opt_g, opt_d], [scheduler_g, scheduler_d]
        
        return [opt_g, opt_d], []


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train a Music Source Restoration Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)

    # Data
    data_module = MusicRestorationDataModule(config.get('data', {}))
    # Model
    model_module = MusicRestorationModule(config)

    # Paths
    project = config.get('project_name', 'msrkit')
    exp_name = config['model']['name']
    exp_name = exp_name.replace(" ", "_")

    save_root = Path(config.get('trainer', {}).get('save_dir', 'runs'))
    save_dir = save_root / project / exp_name
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{step:08d}",
        every_n_train_steps=config.get('trainer', {}).get('checkpoint_save_interval', 10000),
        save_top_k=-1,
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger(s)
    loggers = []

    # TensorBoard (always enabled by default)
    use_tensorboard = config.get('logging', {}).get('tensorboard', {}).get('enabled', True)
    if use_tensorboard:
        tb_logger = TensorBoardLogger(
            save_dir=str(save_root),
            name=project,
            version=exp_name
        )
        loggers.append(tb_logger)

    # WandB (optional)
    wandb_cfg = config.get('logging', {}).get('wandb', {})
    use_wandb = wandb_cfg.get('enabled', False)
    if use_wandb:
        wandb_project = wandb_cfg.get('project', project)
        wandb_name = wandb_cfg.get('name', f"{project}_{exp_name}")
        wandb_entity = wandb_cfg.get('entity', None)
        wandb_tags = wandb_cfg.get('tags', [])
        wandb_notes = wandb_cfg.get('notes', None)

        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_name,
            entity=wandb_entity,
            tags=wandb_tags,
            notes=wandb_notes,
            save_dir=str(save_dir),
            log_model=wandb_cfg.get('log_model', False),
            config=config
        )
        loggers.append(wandb_logger)

    # Use single logger or list of loggers
    logger = loggers[0] if len(loggers) == 1 else loggers
    
    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_steps=config.get('trainer', {}).get('max_steps', 400000),
        log_every_n_steps=config.get('trainer', {}).get('log_every_n_steps', 100),
        devices=config.get('trainer', {}).get('devices', [0]),
        precision=str(config.get('trainer', {}).get('precision', 'bf16-mixed')),
        accelerator="gpu"
    )
    
    trainer.fit(model_module, datamodule=data_module)


if __name__ == '__main__':
    main()