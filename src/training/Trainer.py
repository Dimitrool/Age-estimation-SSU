from pathlib import Path
import torch
import tqdm
import torch.nn as nn
from omegaconf import DictConfig

from src.data.build_data_loader import build_data_loader
from src.constants import CHECKPOINTS_FOLDER_NAME, BEST_CHECKPOINT_NAME, PLOTS_FOLDER_NAME, DATA_DIR
from src.training.EpochAnalyzer import EpochAnalyzer
from src.evaluation.evaluation import evaluate

class Trainer:
    def __init__(self, cfg: DictConfig, model: nn.Module, loss_function, optimizer, scheduler, result_path, device):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trn_loader = build_data_loader(cfg, "training")
        self.val_loader = build_data_loader(cfg, "validation")
        self.test_loader = build_data_loader(cfg, "testing")

        self.accumulation_steps = cfg.training.accumulation_steps
        self.checkpoint_interval = cfg.training.checkpoint_interval
        self.start_epoch = 0
        self.epochs = cfg.training.num_epochs
        self.best_val_loss = float("inf")

        self.step_index = self._get_step_index()
        self.checkpoints_dir = Path(result_path) / CHECKPOINTS_FOLDER_NAME
        self.plots_dir = Path(result_path) / PLOTS_FOLDER_NAME

        self.epoch_analyzer = EpochAnalyzer(result_path, self.scheduler, cfg.log.tensorboard)
        self.device = device

    def train(self):
        self.best_val_loss = float("inf")
        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(self.trn_loader, epoch)
            self.validate_one_epoch(epoch)
        
        self.test_trained_model()
    
    def train_full_dataset(self):
        for epoch in range(self.start_epoch, self.epochs):
            trn_loss = self.train_one_epoch(self.trn_loader, epoch)
            val_loss = self.train_one_epoch(self.val_loader, epoch)
            test_loss = self.train_one_epoch(self.test_loader, epoch)

            loss = (trn_loss + val_loss + test_loss) / 3
            if self.best_val_loss > loss:
                self.best_val_loss = loss
                model_path = self.checkpoints_dir / BEST_CHECKPOINT_NAME
                self._save_training_state(epoch, loss, model_path)
                print(f"New best validation loss: {self.best_val_loss:.5f}")

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        self.step_index = self._get_step_index()

        self.epoch_analyzer.null_epoch_metrics()
        for batch in tqdm(loader, desc="Training"):
            img1, img2, true_age1, true_age2 = batch

            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            true_age1 = true_age1.to(self.device)
            true_age2 = true_age2.to(self.device)

            pred_age2 = self.model(img1, img2, true_age1)

            loss_out = self.loss_function(pred_age2, true_age2)

            self.accumulated_backward(loss_out.loss)
            self.epoch_analyzer.add(loss_out)

        self.finish_train_epoch()
        self.epoch_analyzer.log_epoch("Train", epoch)
        return self.epoch_analyzer.loss

    def validate_one_epoch(self, epoch):
        self.model.eval()

        self.epoch_analyzer.null_epoch_metrics()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                img1, img2, true_age1, true_age2 = batch

                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                true_age1 = true_age1.to(self.device)
                true_age2 = true_age2.to(self.device)

                pred_age2 = self.model(img1, img2, true_age1)

                loss_out = self.loss_function(pred_age2, true_age2)

                self.epoch_analyzer.add(loss_out)

        checkpoint_path = self.checkpoints_dir / f"epoch_{epoch:04d}.pth"
        (self.checkpoints_dir / CHECKPOINTS_FOLDER_NAME).mkdir(exist_ok=True)
        self._save_training_state(epoch, self.epoch_analyzer.loss, checkpoint_path)

        # Keep track of best model and save it in checkpoint format too
        if self.epoch_analyzer.loss < self.best_val_loss:
            self.best_val_loss = self.epoch_analyzer.loss
            model_path = self.checkpoints_dir / BEST_CHECKPOINT_NAME
            self._save_training_state(epoch, self.epoch_analyzer.loss, model_path)
            print(f"New best validation loss: {self.best_val_loss:.5f}")

    def test_trained_model(self):
        results = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                img1, img2, true_age1, true_age2 = batch

                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                true_age1 = true_age1.to(self.device)
                true_age2 = true_age2.to(self.device)

                pred_age2 = self.model(img1, img2, true_age1)

                results.extend([result.item() for result in pred_age2])
         
        evaluate(results, DATA_DIR, self.plots_dir)

    def accumulated_backward(self, loss: torch.Tensor):
        (loss / self.accumulation_steps).backward()
        self.step_index += 1
        if self.step_index % self.accumulation_steps == 0:
            self._flush_grads()

    def finish_train_epoch(self):
        if self.step_index % self.accumulation_steps != 0:
            self._flush_grads()

        self.scheduler.step()

    def _flush_grads(self):
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def load_checkpoint(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.best_val_loss = checkpoint.get("val_loss", float("inf"))

    def _save_training_state(self, epoch, val_loss, path):
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )

    def _get_step_index(self) -> int:
        for st in self.optimizer.state.values():
            if isinstance(st, dict) and "step" in st:
                return int(st["step"])
        return 0
