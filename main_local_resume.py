import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from src.logging.logging import resume_logging
from src.models.get_model import get_checkpoint_path
from src.training.build_trainer import build_trainer

# Import config registration to register structured configs with Hydra
from src.hydra_configs.register_config import register_configs

parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument("results_path", type=str, help="Path to the input JSON file")


def get_config(experiment_dir: Path) -> DictConfig:
    config_path = experiment_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found at {config_path}")

    cfg = DictConfig(OmegaConf.load(config_path))
    return cfg


def resume(args):
    register_configs()
    results_path = Path(args.results_path)

    cfg: DictConfig = get_config(results_path)
    checkpoint_path = get_checkpoint_path(results_path, "latest")
    
    resume_logging(results_path)

    trainer = build_trainer(cfg, results_path, checkpoint_path)
    trainer.train()


if __name__ == "__main__":
    args = parser.parse_args()
    resume(args)

