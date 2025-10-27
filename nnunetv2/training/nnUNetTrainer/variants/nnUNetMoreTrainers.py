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
    Focal Tversky loss (per-class, averaged over batch/classes).

    • Tversky index generalizes Dice by weighting FP (β) and FN (α).
    • Focal term (γ) emphasizes hard examples: (1 - Tversky)^γ.
    • Optional class weights and background exclusion.
    • Optional spatial valid_mask to restrict computation to a region.

    Args:
        alpha (float|Tensor): Weight for FN term (higher → penalize misses).
        beta  (float|Tensor): Weight for FP term (higher → penalize false alarms).
        gamma (float): Focusing power (>1 emphasizes hard examples).
        smooth (float): Small constant for numerical stability.
        class_weights (Tensor|None): Per-class weights (C,) applied after focal.
        ignore_background (bool): Drop channel 0 from reduction if C>1.

    Inputs:
        logits (B, C, ...): Unnormalized class scores.
        target_onehot (B, C, ...): One-hot segmentation labels.
        valid_mask (optional): (B,1,...) or (B,C,...) spatial mask.

    Returns:
        Scalar tensor: mean focal Tversky loss across (non-background) classes.
    """

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

    def _as_param(self, x, ref: torch.Tensor) -> torch.Tensor | None:
        """
        Ensure α or β are tensors on the same device/dtype as 'ref' and 2D-shaped (1, C or 1).
        Accepts scalars or tensors. Returns None if x is neither.
        """
        if isinstance(x, torch.Tensor):
            return x.to(ref.device, ref.dtype).view(1, -1)
        if isinstance(x, (int, float)):
            return torch.tensor(x, device=ref.device, dtype=ref.dtype).view(1, 1)
        return None

    def forward(
        self,
        logits: torch.Tensor,
        target_onehot: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Convert logits to probabilities
        p = torch.softmax(logits, dim=1)  # (B, C, ...)
        dims = tuple(range(2, p.ndim))  # spatial dimensions to reduce over

        # Apply optional spatial mask: restricts computation to ROI
        if valid_mask is not None:
            vm = valid_mask
            # Broadcast (B,1,...) → (B,C,...) if needed
            if vm.ndim == p.ndim - 1:  # (B,1,...)
                vm = vm.expand(-1, p.shape[1], *([-1] * (p.ndim - 2)))
            p_ = p * vm
            y_ = target_onehot * vm
            invy_ = (1 - target_onehot) * vm
        else:
            p_, y_, invy_ = p, target_onehot, (1 - target_onehot)

        # Per-class TP/FP/FN
        tp = (p_ * y_).sum(dims)  # (B, C)
        fp = (p_ * invy_).sum(dims)  # (B, C)
        fn = ((1 - p_) * y_).sum(dims)  # (B, C)

        # Allow α, β to be scalars or per-class tensors
        alpha = self._as_param(self.alpha, tp)  # (1, C) or (1,1)
        beta = self._as_param(self.beta, tp)  # (1, C) or (1,1)

        # Tversky index per class: (TP + s) / (TP + α·FN + β·FP + s)
        t = (tp + self.smooth) / (
            tp + (alpha * fn) + (beta * fp) + self.smooth
        )  # (B, C)

        # Focal Tversky loss per class
        ft = (1.0 - t).clamp_min(0) ** self.gamma  # (B, C)

        # Optionally remove background channel (assumed channel 0)
        if self.ignore_background and ft.size(1) > 1:
            ft = ft[:, 1:]
            # Prepare class weights accordingly (drop bg)
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

        # Optional per-class weighting (normalize to mean=1 for scale stability)
        if cw is not None:
            cw = cw / (cw.mean() + 1e-8)
            ft = ft * cw.view(1, -1)

        # Mean over remaining classes and batch
        return ft.mean()


class FocalTverskyLossWrapper(nn.Module):
    """
    Composite segmentation loss: FocalTversky + 0.2 * CrossEntropy.

    • FocalTversky (α, β, γ) focuses on class overlap with FN/FP trade-off
      and focal modulation; optionally ignores background and supports ROI masks.
    • Cross-Entropy complements with calibration on logits and class counts.
    • Accepts either a target dict ({"target","onehot","mask"}) or a one-hot tensor.

    Args:
        ce_ignore_index (int): Label to ignore in CE (e.g., padding/void).
    """

    def __init__(self, ce_ignore_index: int = -1):
        super().__init__()
        self.ft = FocalTverskyLoss(
            alpha=0.85, beta=0.15, gamma=1.33, ignore_background=True
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)

    def _one(self, logits: torch.Tensor, target) -> torch.Tensor:
        """
        Compute composite loss for a single prediction head.

        Handles:
          • dict target: y (indices), yoh (one-hot), and optional vm (mask)
          • one-hot target: infers y via argmax; no mask
        """
        if isinstance(target, dict):
            y, yoh = target["target"], target["onehot"]
            vm = target.get("mask", None)
        else:
            yoh = target
            y = torch.argmax(yoh, dim=1)
            vm = None

        # Focal Tversky on probabilities (with optional ROI mask) + CE on logits
        return self.ft(logits, yoh, valid_mask=vm) + 0.2 * self.ce(logits, y)

    def forward(self, logits, target) -> torch.Tensor:
        """
        Supports deep supervision:
          • If logits is a list/tuple of heads, apply per-head loss with weights:
            w0=1.0 for main head, 0.5 for each auxiliary head, then average.
          • Else compute single-head loss.
        """
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

    def train_step(self, batch: dict) -> dict:
        """
        Override train_step to build target dictionary for custom loss wrappers.
        Converts raw target tensor to dict with 'target', 'onehot', 'mask'.
        """
        from torch.cuda.amp import autocast
        from contextlib import nullcontext as dummy_context

        data = batch["data"]
        target_raw = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target_raw, list):
            target_raw = [i.to(self.device, non_blocking=True) for i in target_raw]
        else:
            target_raw = target_raw.to(self.device, non_blocking=True)

        # Build target dictionary for loss function
        if isinstance(target_raw, list):
            # Deep supervision: list of targets at different scales
            target_dict = target_raw
        else:
            # Single scale: build dictionary
            target_dict = {
                "target": target_raw,
                "onehot": self._convert_to_onehot(target_raw),
            }

            # Add mask if ignore label is defined
            if self.label_manager.has_ignore_label:
                target_dict["mask"] = (
                    target_raw != self.label_manager.ignore_label
                ).float()

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            loss_val = self.loss(output, target_dict)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_val).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss_val.detach().cpu().numpy()}

    @staticmethod
    def _convert_to_onehot(target: torch.Tensor) -> torch.Tensor:
        """
        Convert long tensor to one-hot encoding.
        Args:
            target: (B, ...) long tensor with class indices
        Returns:
            (B, C, ...) one-hot encoded tensor
        """
        if target.ndim < 2:
            target = target.unsqueeze(0)

        # Get number of classes from max value
        num_classes = int(target.max().item()) + 1

        # Create one-hot encoding
        onehot = torch.zeros(
            (target.shape[0], num_classes, *target.shape[1:]),
            dtype=torch.float32,
            device=target.device,
        )
        onehot.scatter_(1, target.long().unsqueeze(1), 1.0)
        return onehot


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
        # Get ignore label from label manager if it exists
        ignore_idx: int = (
            self.label_manager.ignore_label  # type: ignore
            if self.label_manager.has_ignore_label
            else -1
        )
        return FocalTverskyLossWrapper(ce_ignore_index=ignore_idx)


class TopKCrossEntropy(nn.Module):
    """
    Top-k Cross-Entropy (per-sample).

    Computes pixel/voxel-wise cross-entropy, but for each sample in the batch
    only averages over the hardest k% valid locations (largest CE values).
    Positions marked with `ignore_index` are excluded from the top-k selection.

    Args:
        k_ratio (float): Fraction (0–1] of *valid* voxels per sample to keep.
                         E.g., 0.2 keeps the top 20% highest CE values.
        ignore_index (int): Class label to ignore in the loss. If >= 0, those
                            positions are excluded from the top-k selection.

    Inputs:
        logits (FloatTensor): shape (B, C, ...) class scores (unnormalized).
        target_long (LongTensor): shape (B, ...) integer class labels.

    Returns:
        loss (Tensor): scalar mean over the batch of per-sample top-k means.

    Notes:
        • If a sample has very few valid voxels (e.g., tiny masks), the code
          gracefully falls back to averaging over *all* valid voxels.
        • If a sample has no valid voxels at all, it falls back to the mean of
          the raw CE tensor (which is 0 at ignored positions under PyTorch’s
          `ignore_index` semantics), thus yielding 0 contribution.
        • This is useful for highly imbalanced segmentation where easy background
          can dominate the loss; focusing on the hardest k% emphasizes hard/edge
          regions and typically stabilizes early training.
    """

    def __init__(self, k_ratio=0.2, ignore_index: int = -1):
        super().__init__()
        self.k_ratio = float(k_ratio)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target_long: torch.Tensor) -> torch.Tensor:
        # Compute unreduced CE at each spatial position. PyTorch returns 0 at
        # positions equal to `ignore_index` when reduction="none".
        # shapes: logits (B, C, ...), target_long (B, ...)
        ce = F.cross_entropy(
            logits, target_long, reduction="none", ignore_index=self.ignore_index
        )  # (B, ...)

        B = ce.shape[0]
        ce_flat = ce.view(B, -1)  # Flatten spatial dims → (B, N)

        # Build a boolean mask of valid positions (True=include, False=ignore).
        if self.ignore_index is not None and self.ignore_index >= 0:
            valid = target_long.view(B, -1) != self.ignore_index  # (B, N)
        else:
            valid = torch.ones_like(ce_flat, dtype=torch.bool)  # (B, N)

        # Prevent ignored positions from being selected by topk by setting them
        # to -inf in a masked view (topk will never pick -inf when k < #valid).
        ce_masked = ce_flat.masked_fill(~valid, float("-inf"))  # (B, N)

        # Compute how many positions to keep per sample (at least 1).
        # k_per[b] = ceil(k_ratio * #valid_b), but implemented via clamp_min then long.
        k_per = (valid.sum(dim=1).float() * self.k_ratio).clamp_min(1).long()  # (B,)

        vals = []
        for b in range(B):
            # Number of valid positions in sample b
            n_valid_b = int(valid[b].sum().item())
            k = int(k_per[b])

            if n_valid_b == 0:
                # No valid voxels: fallback to mean over raw CE (which will be 0
                # if all were ignored by PyTorch semantics).
                v = ce_flat[b].mean()
            elif k >= n_valid_b:
                # If k is greater/equal to the number of valid voxels,
                # average over all valid voxels (stable for tiny objects).
                v = ce_flat[b][valid[b]].mean()
            else:
                # Typical case: take the top-k hardest valid voxels and average.
                topk_vals, _ = torch.topk(ce_masked[b], k, largest=True, sorted=False)
                v = topk_vals.mean()

            vals.append(v)

        # Return the batch mean of per-sample values (scalar)
        return torch.stack(vals).mean()


class TopKCELossWrapper(nn.Module):
    """
    Composite loss for segmentation: Top-K Cross-Entropy + 0.5 * Soft Dice.

    • Top-K CE focuses the CE term on the hardest k% valid voxels per sample.
    • Soft Dice stabilizes learning on small/imbalanced structures.
    • Optional `ignore_index` for CE and optional spatial `valid_mask` for Dice.

    Expected targets:
      - Dict form (typical train_step):
          target["target"] : LongTensor (B, ...)       class indices
          target["onehot"] : FloatTensor (B, C, ...)   one-hot labels
          target["mask"]   : (optional) (B,1,...) or (B,C,...) boolean/float
        OR
      - One-hot tensor (B, C, ...). In this case, class indices are inferred
        via argmax along channel dimension.

    Returns:
        Scalar loss = TopKCE(logits, y) + 0.5 * SoftDice(logits, onehot[, mask])
    """

    def __init__(self, k_ratio: float = 0.2, ce_ignore_index: int = -1):
        super().__init__()
        self.topk = TopKCrossEntropy(k_ratio=k_ratio, ignore_index=ce_ignore_index)

    def forward(self, logits: torch.Tensor, target) -> torch.Tensor:
        """
        Args:
            logits: FloatTensor (B, C, ...) unnormalized scores.
            target: dict with keys {"target","onehot",["mask"]} or one-hot tensor.
        """
        # Accept either dict targets or raw one-hot targets
        if isinstance(target, dict):
            y = target["target"]  # (B, ...)
            yoh = target["onehot"]  # (B, C, ...)
            vm = target.get("mask", None)  # optional (B,1,...) or (B,C,...)
        else:
            # If only one-hot is provided (e.g., DeepSupervisionWrapper)
            yoh = target  # (B, C, ...)
            y = torch.argmax(yoh, dim=1)  # (B, ...)
            vm = None

        # Hard-example mining CE
        topk_loss = self.topk(logits, y)

        # Region-aware overlap loss
        dice_loss = self._soft_dice(logits, yoh, valid_mask=vm)

        # Composite
        return topk_loss + 0.5 * dice_loss

    @staticmethod
    def _soft_dice(
        logits: torch.Tensor,
        target_onehot: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Soft Dice over channels, with optional spatial masking.

        Args:
            logits        : (B, C, ...)
            target_onehot : (B, C, ...)
            valid_mask    : optional (B,1,...) or (B,C,...) (bool/float).
                            When provided, Dice is computed only over masked areas.

        Returns:
            Mean Dice loss across batch and (non-background) classes.
        """
        # Convert logits to probabilities
        p = torch.softmax(logits, dim=1)  # (B, C, ...)

        if valid_mask is not None:
            vm = valid_mask
            # If mask lacks channel dim, broadcast to channels
            if vm.ndim == p.ndim - 1:  # (B, 1, ...)
                vm = vm.expand(-1, p.shape[1], *([-1] * (p.ndim - 2)))
            # Ensure float mask for safe multiplication
            if vm.dtype != p.dtype:
                vm = vm.to(dtype=p.dtype)
            p = p * vm
            target_onehot = target_onehot * vm

        # Sum over spatial dims only
        dims = tuple(range(2, p.ndim))
        num = 2 * (p * target_onehot).sum(dims)  # (B, C)
        den = (p + target_onehot).sum(dims).clamp_min(eps)  # (B, C)

        dice = 1 - (num / den)  # per-class dice loss (B, C)

        # Drop background channel if present
        if dice.size(1) > 1:
            dice = dice[:, 1:]

        # Mean over classes and batch
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
        # Get ignore label from label manager if it exists
        ignore_idx: int = (
            self.label_manager.ignore_label  # type: ignore
            if self.label_manager.has_ignore_label
            else -1
        )
        return FocalTverskyLossWrapper(ce_ignore_index=ignore_idx)


class EmaDiceTopKCETrainer(BaseEmaEarlyStopTrainer, BaseLossTrainer):
    """
    EMA early stopping + Top-K CE loss.
    """

    def _build_loss_module(self) -> nn.Module:
        # Get ignore label from label manager if it exists
        ignore_idx: int = (
            self.label_manager.ignore_label  # type: ignore
            if self.label_manager.has_ignore_label
            else -1
        )
        return TopKCELossWrapper(ce_ignore_index=ignore_idx)
