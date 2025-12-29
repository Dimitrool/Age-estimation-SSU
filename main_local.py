import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.constants import CONFIGS_PATH, CONFIG_TEMPLATE_NAME
from src.utils.logging import configure_logging


@hydra.main(config_path=str(CONFIGS_PATH), config_name=CONFIG_TEMPLATE_NAME, version_base=None)
def main(cfg: DictConfig):
    configure_logging()


if __name__ == "__main__":
    main()
