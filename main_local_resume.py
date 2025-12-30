import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import cast

from src.logging.logging import resume_logging
from src.models.get_model import get_checkpoint_path
from src.training.build_trainer import build_trainer


parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument("results_path", type=str, help="Path to the input JSON file")


def get_config(experiment_dir: Path) -> DictConfig:
    config_path = experiment_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found at {config_path}")

    cfg = cast(DictConfig, OmegaConf.load(config_path))
    return cfg["config"]


def resume(args):
    results_path = Path(args.results_path)

    cfg: DictConfig = get_config(results_path)
    checkpoint_path = get_checkpoint_path(results_path, "latest")
    
    resume_logging(results_path)

    trainer = build_trainer(cfg, results_path, checkpoint_path)
    trainer.train()


if __name__ == "__main__":
    args = parser.parse_args()
    resume(args)

