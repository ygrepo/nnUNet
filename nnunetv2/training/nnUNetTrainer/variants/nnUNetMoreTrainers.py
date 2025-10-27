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
    def __init__(
        self,
        alpha=0.85,
        beta=0.15,
        gamma=1.33,
        smooth=1e-6,
        class_weights=None,
        ignore_background=True,
    ):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.gamma, self.smooth = gamma, smooth
        self.class_weights = class_weights
        self.ignore_background = ignore_background

    def _as_param(self, x, ref):
        if isinstance(x, torch.Tensor):
            return x.to(ref.device, ref.dtype).view(1, -1)
        if isinstance(x, (int, float)):
            return torch.tensor(x, device=ref.device, dtype=ref.dtype).view(1, 1)
        return None

    def forward(self, logits, target_onehot, valid_mask=None):
        p = torch.softmax(logits, dim=1)  # (B,C,...)
        dims = tuple(range(2, p.ndim))

        if valid_mask is not None:
            vm = valid_mask
            if vm.ndim == p.ndim - 1:  # (B,1,...)
                vm = vm.expand(-1, p.shape[1], *([-1] * (p.ndim - 2)))
            p_ = p * vm
            y_ = target_onehot * vm
            invy_ = (1 - target_onehot) * vm
        else:
            p_, y_, invy_ = p, target_onehot, (1 - target_onehot)

        tp = (p_ * y_).sum(dims)
        fp = (p_ * invy_).sum(dims)
        fn = ((1 - p_) * y_).sum(dims)

        alpha = self._as_param(self.alpha, tp)
        beta = self._as_param(self.beta, tp)
        t = (tp + self.smooth) / (tp + alpha * fn + beta * fp + self.smooth)  # (B,C)
        ft = (1.0 - t).clamp_min(0) ** self.gamma

        if self.ignore_background and ft.size(1) > 1:
            ft = ft[:, 1:]
            cw = (
                None
                if self.class_weights is None
                else self.class_weights.to(ft.device, ft.dtype)[1:]
            )
        else:
            cw = (
                None
                if self.class_weights is None
                else self.class_weights.to(ft.device, ft.dtype)
            )

        if cw is not None:
            cw = cw / (cw.mean() + 1e-8)
            ft = ft * cw.view(1, -1)

        return ft.mean()


class FocalTverskyLossWrapper(nn.Module):
    def __init__(self, ce_ignore_index=-1):
        super().__init__()
        self.ft = FocalTverskyLoss(
            alpha=0.85, beta=0.15, gamma=1.33, ignore_background=True
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)

    def _one(self, logits, target):
        # Handle both dict target (from train_step) and tensor target
        # (from DeepSupervisionWrapper)
        if isinstance(target, dict):
            y, yoh = target["target"], target["onehot"]
            vm = target.get("mask", None)
        else:
            # When called by DeepSupervisionWrapper, target is one-hot tensor
            yoh = target
            y = torch.argmax(yoh, dim=1)
            vm = None
        return self.ft(logits, yoh, valid_mask=vm) + 0.2 * self.ce(logits, y)

    def forward(self, logits, target):
        if isinstance(logits, (list, tuple)):
            weights = [1.0] + [0.5] * (len(logits) - 1)
            s = sum(w * self._one(lg, target) for w, lg in zip(weights, logits))
            return s / sum(weights)
        return self._one(logits, target)


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
    Recommended: crank up FG patch oversampling for tiny lesions.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Increase foreground oversampling for better recall on small targets
        self.oversample_foreground_percent = 0.9

    def _build_loss_module(self) -> nn.Module:
        return FocalTverskyLossWrapper()


class TopKCrossEntropy(nn.Module):
    """
    CE averaged over the hardest k% voxels per sample, with ignore support.
    """

    def __init__(self, k_ratio=0.2, ignore_index: int = -1):
        super().__init__()
        self.k_ratio = float(k_ratio)
        self.ignore_index = ignore_index

    def forward(self, logits, target_long):
        # logits: (B,C,...) ; target_long: (B, ...)
        ce = F.cross_entropy(
            logits, target_long, reduction="none", ignore_index=self.ignore_index
        )  # (B, ...)
        B = ce.shape[0]
        ce_flat = ce.view(B, -1)  # (B, N)
        # Mask out ignored positions (equal to 0 loss from CE w/ ignore_index)
        # But we still must exclude them from the top-k selection.
        if self.ignore_index is not None and self.ignore_index >= 0:
            # Recompute a mask from target_long (1=valid,0=ignore)
            valid = target_long.view(B, -1) != self.ignore_index
        else:
            valid = torch.ones_like(ce_flat, dtype=torch.bool)

        # For invalid voxels, set loss to -inf so they are never chosen by topk
        ce_masked = ce_flat.masked_fill(~valid, float("-inf"))

        # choose k per sample based on number of valid voxels
        k_per = (valid.sum(dim=1).float() * self.k_ratio).clamp_min(1).long()
        vals = []
        for b in range(B):
            # if very few valid voxels, fallback to mean over valid
            k = int(k_per[b])
            if k >= valid[b].sum().item():
                v = ce_flat[b][valid[b]].mean() if valid[b].any() else ce_flat[b].mean()
            else:
                topk_vals, _ = torch.topk(ce_masked[b], k, largest=True, sorted=False)
                v = topk_vals.mean()
            vals.append(v)
        return torch.stack(vals).mean()


class TopKCELossWrapper(nn.Module):
    """Top-K CE + 0.5 * Soft Dice, with ignore support."""

    def __init__(self, k_ratio=0.2, ce_ignore_index=-1):
        super().__init__()
        self.topk = TopKCrossEntropy(k_ratio=k_ratio, ignore_index=ce_ignore_index)

    def forward(self, logits, target):
        # Handle both dict target (from train_step) and tensor target
        # (from DeepSupervisionWrapper)
        if isinstance(target, dict):
            y = target["target"]  # (B, ...)
            yoh = target["onehot"]  # (B, C, ...)
            vm = target.get("mask", None)  # optional (B,1,...) or (B,C,...)
        else:
            # When called by DeepSupervisionWrapper, target is one-hot
            yoh = target
            y = torch.argmax(yoh, dim=1)
            vm = None

        topk_loss = self.topk(logits, y)
        dice_loss = self._soft_dice(logits, yoh, valid_mask=vm)
        return topk_loss + 0.5 * dice_loss

    @staticmethod
    def _soft_dice(logits, target_onehot, valid_mask=None, eps=1e-6):
        p = torch.softmax(logits, dim=1)  # (B,C,...)
        if valid_mask is not None:
            vm = valid_mask
            if vm.ndim == p.ndim - 1:  # (B,1,...)
                vm = vm.expand(-1, p.shape[1], *([-1] * (p.ndim - 2)))
            p = p * vm
            target_onehot = target_onehot * vm
        dims = tuple(range(2, p.ndim))
        num = 2 * (p * target_onehot).sum(dims)
        den = (p + target_onehot).sum(dims) + eps
        dice = 1 - (num / den)  # (B,C)
        if dice.size(1) > 1:  # drop background
            dice = dice[:, 1:]
        return dice.mean()


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
    patience: int = 30
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
