import hydra
import pickle
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.constants import CONFIGS_PATH, CONFIG_TEMPLATE_NAME
from src.logging.logging import configure_logging
from src.training.build_trainer import build_trainer

# Import config registration to register structured configs with Hydra
import src.hydra_configs.register_config  # noqa: F401


@hydra.main(config_path=str(CONFIGS_PATH), config_name=CONFIG_TEMPLATE_NAME, version_base=None)
def main(cfg: DictConfig):
    configure_logging()

    hydra_cfg = HydraConfig.get()
    results_path = Path(hydra_cfg.runtime.output_dir)

    trainer = build_trainer(cfg, results_path)

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # 2. Save as pickle
    name = f"{cfg.experiment_name}.pkl"
    with open(Path("pickle_configs", name), "wb") as f:
        pickle.dump(config_dict, f)

    if cfg.training.use_full_dataset:
        trainer.train_full_dataset()
    else:
        trainer.train()

if __name__ == "__main__":
    main()
