import torch
from omegaconf import DictConfig

from src.data.build_data_loader import build_data_loader


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function

        self.trn_loader = build_data_loader(cfg, "training")
        self.val_loader = build_data_loader(cfg, "validation")


