from pathlib import Path

import torch
from omegaconf import DictConfig

import src.training.loss_functions as ls
from src.models.get_model import get_model
from src.training.Trainer import Trainer


LOSS_REGISTRY = {
    "custom": ls.CustomLoss,
    "l2": ls.MSELoss,
    "mse": ls.MSELoss,
    "l1": ls.L1Loss,
    "mae": ls.L1Loss,
}


def build_trainer(cfg: DictConfig, results_path: Path, checkpoint_path: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg, device)
    loss_function = _build_loss_function(cfg.loss)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    trainer = Trainer(
        cfg = cfg,
        model = model,
        loss_function = loss_function,
        optimizer = optimizer,
        scheduler = scheduler,
        result_path = results_path,
        device = device
    )

    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path, device)

    return trainer


def _build_loss_function(cfg_loss):
    name = str(cfg_loss.type).lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}")
    return LOSS_REGISTRY[name]()


def _build_optimizer(cfg: DictConfig, model):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )
    return optimizer


def _build_scheduler(cfg: DictConfig, optimizer):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.optimizer.scheduler_gamma
    )
    return scheduler

