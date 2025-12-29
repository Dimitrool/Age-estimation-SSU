from pathlib import Path
from typing import Union
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from src.training.loss_functions import LossOutput

LRScheduler = Union[_LRScheduler, ReduceLROnPlateau]


class EpochAnalyzer:
    def __init__(self, log_dir: Path, scheduler: LRScheduler, use_tb: bool):
        self.writer = SummaryWriter(log_dir=log_dir / "Tensorboard")
        self.use_tb = use_tb
        self.scheduler = scheduler

        self.null_epoch_metrics()

    def null_epoch_metrics(self):
        self.loss = 0.0
        self.steps = 0

    def add(self, loss_output: LossOutput):
        self.loss += float(loss_output.loss.item())
        self.steps += 1

    def log_epoch(self, split, epoch):
        self._finalize()

        msg = f"[Epoch {epoch+1:03d}] {split} Loss={self.loss:.5f}"
        print(msg)

        if self.use_tb:
            self._write_metrics_to_tensorboard(split, epoch)

    def _finalize(self):
        if self.steps == 0:
            self.loss = float("inf")
        else:
            self.loss =          self.loss / self.steps

    def _write_metrics_to_tensorboard(self, split, epoch):
        self.writer.add_scalar(f"{split}/Loss",    self.loss, epoch)

        if split == "Train":
            lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, "get_last_lr") else self.scheduler.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LR", lr, epoch)

