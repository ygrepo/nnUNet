# EmaDiceEarlyStopTrainer.py
from __future__ import annotations
import math
from os.path import join
from typing import Any, Optional

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class EmaDiceEarlyStopTrainer(nnUNetTrainer):
    """
    Early-stop on EMA-smoothed validation Dice (higher is better).
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

    # ---- helpers ----
    @staticmethod
    def _extract_val_dice(val_out: Any) -> float:
        """
        Try common keys nnU-Net v2 returns. Falls back to NaN-safe behavior.
        """
        if isinstance(val_out, dict):
            # try typical keys
            for k in (
                "meanDice",
                "mean_dice",
                "aggregated_dice",
                "dice",
                "global_dice",
            ):
                if k in val_out:
                    return float(val_out[k])
            # some versions expose classwise stats; average them if present
            if "dice_per_class" in val_out and isinstance(
                val_out["dice_per_class"], (list, tuple)
            ):
                arr = np.asarray(val_out["dice_per_class"], dtype=float)
                return float(np.nanmean(arr))
            if "dice_per_class_and_case" in val_out:
                arr = np.asarray(val_out["dice_per_class_and_case"], dtype=float)
                return float(np.nanmean(arr))
        # scalar?
        try:
            return float(val_out)
        except Exception:
            return float("nan")

    def _run_validation(self) -> float:
        """Run validation and extract Dice score."""
        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):
                val_step = self.validation_step(next(self.dataloader_val))
                val_outputs.append(val_step)
            self.on_validation_epoch_end(val_outputs)

        # Extract mean Dice from logger
        dice_values = self.logger.my_fantastic_logging.get(
            "dice_per_class_or_region", [[]]
        )
        if dice_values:
            d = float(np.nanmean(dice_values[-1]))
        else:
            d = float("nan")

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
                checkpoint_path = join(self.output_folder, "checkpoint_best.pth")
                self.save_checkpoint(checkpoint_path)
            except Exception:
                pass
            msg = "[EMA] Improvement — checkpoint saved."
            self.print_to_log_file(msg)
        else:
            if self._val_calls > self.warmup_validations:
                self.bad_count += 1
                if self.bad_count >= self.patience:
                    msg = (
                        f"[EMA] Patience {self.patience} exhausted. " "Early stopping."
                    )
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
