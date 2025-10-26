# my_trainers/focal_tversky_trainer.py
from __future__ import annotations
import math
from os.path import join
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class FocalTverskyLoss(nn.Module):
    """
    Multi-class Focal Tversky loss
    params:
      alpha: weight for FN (↑alpha => penalize FN)
      beta : weight for FP
      gamma: focal focusing (>=1), e.g., 1.33–2.0
      smooth: epsilon
    expects: logits (B,C,...) and target one-hot (B,C,...)
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target_onehot):
        probs = torch.softmax(logits, dim=1)
        dims = tuple(range(2, probs.ndim))

        tp = (probs * target_onehot).sum(dims)
        fp = (probs * (1 - target_onehot)).sum(dims)
        fn = ((1 - probs) * target_onehot).sum(dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        ft = torch.pow(1.0 - tversky, self.gamma)  # (B,C)
        # ignore background (c=0) if your labels use 0 as bg
        ft = ft[:, 1:] if ft.shape[1] > 1 else ft
        return ft.mean()


class TopKCrossEntropy(nn.Module):
    """
    Top-K CE: average CE over the hardest k% voxels.
    k_ratio in (0,1] e.g., 0.2
    """

    def __init__(self, k_ratio=0.2):
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, logits, target_long):
        ce = F.cross_entropy(logits, target_long, reduction="none")  # (B, ...)
        ce_flat = ce.view(ce.shape[0], -1)
        k = ce_flat.shape[1] * self.k_ratio
        k = max(1, int(k))
        topk_vals, _ = torch.topk(ce_flat, k, dim=1)
        return topk_vals.mean()


class FocalTverskyLossWrapper(nn.Module):
    """Wrapper to combine Focal Tversky and CE loss."""

    def __init__(self):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.5, smooth=1e-6)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        # target is dict with 'target' (long) and 'onehot' (float)
        y = target["target"]
        yoh = target["onehot"]
        return self.ft(logits, yoh) + 0.2 * self.ce(logits, y)


class BaseLossTrainer(nnUNetTrainer):
    """Base class for trainers that only override the loss function."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _build_loss_module(self) -> nn.Module:
        """
        Subclasses should override this to return their custom loss module.
        """
        msg = "Subclasses must implement _build_loss_module()"
        raise NotImplementedError(msg)

    def _build_loss(self):
        loss = self._build_loss_module()

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            if deep_supervision_scales is not None:
                weights = np.array(
                    [1 / (2**i) for i in range(len(deep_supervision_scales))]
                )
                weights[-1] = 0
                weights = weights / weights.sum()
                loss = DeepSupervisionWrapper(loss, weights)

        return loss


class FocalTverskyTrainer(BaseLossTrainer):
    """
    Replace the default DC+CE with Focal Tversky (optionally + small CE).
    """

    def _build_loss_module(self) -> nn.Module:
        return FocalTverskyLossWrapper()


class TopKCELossWrapper(nn.Module):
    """Wrapper to combine Top-K CE and Soft Dice loss."""

    def __init__(self):
        super().__init__()
        self.topk = TopKCrossEntropy(k_ratio=0.2)

    def forward(self, logits, target):
        y = target["target"]
        yoh = target["onehot"]
        topk_loss = self.topk(logits, y)
        dice_loss = self._soft_dice(logits, yoh)
        return topk_loss + 0.5 * dice_loss

    @staticmethod
    def _soft_dice(logits, target_onehot, eps=1e-6):
        p = torch.softmax(logits, dim=1)
        dims = tuple(range(2, p.ndim))
        num = 2 * (p * target_onehot).sum(dims)
        den = (p + target_onehot).sum(dims) + eps
        dice = 1 - (num / den)
        dice = dice[:, 1:] if dice.shape[1] > 1 else dice
        return dice.mean()


class TopKCETrainer(BaseLossTrainer):
    """
    Replace CE with Top-K CE (optionally + Dice).
    """

    def _build_loss_module(self) -> nn.Module:
        return TopKCELossWrapper()


class BaseEmaEarlyStopTrainer(nnUNetTrainer):
    """
    Base class for trainers with EMA-based early stopping.
    - Tracks EMA(mean Dice) with smoothing factor alpha
    - Saves checkpoint when EMA improves by >= min_delta
    - Stops after `patience` validations without improvement
    """

    # tune these:
    # EMA smoothing factor in (0,1]; larger = less smoothing
    alpha: float = 0.2
    # validations without EMA improvement
    patience: int = 20
    # minimal EMA improvement to reset patience
    min_delta: float = 1e-4
    warmup_validations: int = (
        0  # number of validations to ignore for early-stop counting
    )
    validate_every: int = 1  # run validation every N epochs

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 400
        self.best_ema: float = -math.inf
        self.ema: Optional[float] = None
        self.bad_count: int = 0
        self._val_calls: int = 0

    def _run_validation(self) -> float:
        """Run validation and extract Dice score."""
        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for _ in range(self.num_val_iterations_per_epoch):
                val_step = self.validation_step(next(self.dataloader_val))
                val_outputs.append(val_step)
            self.on_validation_epoch_end(val_outputs)

        # Extract mean Dice from logger (logged by on_validation_epoch_end)
        dice_per_class = self.logger.my_fantastic_logging.get(
            "dice_per_class_or_region", [[]]
        )[-1]
        d = float(np.nanmean(dice_per_class))

        if not np.isfinite(d):
            self.print_to_log_file(
                "[EMA] Validation Dice not finite; treating as -inf."
            )
            d = -math.inf
        return d

    def _update_ema(self, x: float) -> float:
        if self.ema is None or not np.isfinite(self.ema):
            self.ema = x
        else:
            self.ema = self.alpha * x + (1.0 - self.alpha) * self.ema
        return self.ema

    def _maybe_validate_and_early_stop(self, epoch: int) -> bool:
        if (epoch + 1) % self.validate_every != 0:
            return False

        dice = self._run_validation()
        ema = self._update_ema(dice)
        self._val_calls += 1

        self.print_to_log_file(
            f"[EMA] epoch={epoch:03d} dice={dice:.6f} ema={ema:.6f} "
            f"best_ema={self.best_ema:.6f} bad={self.bad_count}"
        )

        improved = (ema - self.best_ema) >= self.min_delta
        if improved:
            self.best_ema = ema
            # reset patience and save a checkpoint tagged as best
            self.bad_count = 0
            try:
                path = join(self.output_folder, "checkpoint_best.pth")
                self.save_checkpoint(path)
            except Exception:
                pass
            msg = "[EMA] Improvement — checkpoint saved."
            self.print_to_log_file(msg)
        else:
            if self._val_calls > self.warmup_validations:
                self.bad_count += 1
                if self.bad_count >= self.patience:
                    msg = f"[EMA] Patience {self.patience} exhausted."
                    msg += " Early stopping."
                    self.print_to_log_file(msg)
                    return True
        return False

    def run_training(self):
        msg = (
            f"Starting training with EMA early stopping "
            f"(alpha={self.alpha}, patience={self.patience}, "
            f"min_delta={self.min_delta})"
        )
        self.print_to_log_file(msg)
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_step = self.train_step(next(self.dataloader_train))
                train_outputs.append(train_step)
            self.on_train_epoch_end(train_outputs)

            if self._maybe_validate_and_early_stop(epoch):
                break

            self.on_epoch_end()

        self.on_train_end()
        self.print_to_log_file("Training finished (EMA early stop).")


class EmaDiceFocalTverskyTrainer(BaseEmaEarlyStopTrainer, BaseLossTrainer):
    """
    EMA early stopping + Focal Tversky loss.
    """

    def _build_loss_module(self) -> nn.Module:
        return FocalTverskyLossWrapper()


class EmaDiceTopKCETrainer(BaseEmaEarlyStopTrainer, BaseLossTrainer):
    """
    EMA early stopping + Top-K CE loss.
    """

    def _build_loss_module(self) -> nn.Module:
        return TopKCELossWrapper()
