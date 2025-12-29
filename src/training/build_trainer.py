from pathlib import Path

import torch
from omegaconf import DictConfig

import src.training.loss_functions as ls
from windpredictor.models.build_model import build_model
from windpredictor.training.Trainer import Trainer

LOSS_REGISTRY = {
    "l1": ls.L1Loss,
    "mae": ls.L1Loss,
}


def build_trainer(cfg: DictConfig, results_path: Path, checkpoint_path: Path | None = None):
    model = build_model(cfg)
    loss_function = _build_loss_function(cfg.loss, cfg.dataset.use_tke)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    trainer = Trainer(
        cfg = cfg,
        log_dir = results_path,
        model = model,
        loss_function = loss_function,
        optimizer = optimizer,
        scheduler = scheduler,
    )

    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)

    return trainer


def _build_loss_function(cfg_loss, use_tke):
    name = str(cfg_loss.type).lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}")
    if name == "beta_nll":
        beta = float(cfg_loss.get("beta", 0.5))
        return LOSS_REGISTRY[name](use_tke=use_tke, beta=beta)
    if name == "evidential":
        lambda_reg = float(cfg_loss.get("lambda_reg", 0.01))
        return LOSS_REGISTRY[name](use_tke=use_tke, lambda_reg=lambda_reg)
    return LOSS_REGISTRY[name](use_tke=use_tke)


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

