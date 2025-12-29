import src.hydra_configs.declare_config as hc
from hydra.core.config_store import ConfigStore

from src.constants import CONFIG_TEMPLATE_NAME



def register_configs() -> None:
    """
    Registers the configuration schema with Hydra's ConfigStore.
    Call this function at the top of your train.py before @hydra.main.
    """
    cs = ConfigStore.instance()

    # Registering 'base_config' as the primary node name allows 
    # your YAML file to validate against this schema.
    cs.store(name=CONFIG_TEMPLATE_NAME, node=hc.BaseConfig)

    # Register data configs
    cs.store(group="data", name="base_data", node=hc.DataConfig)
    cs.store(group="data/augmentations", name="base_augmentations", node=hc.AugmentationsConfig)

    # Register architecture configs
    cs.store(group="architecture", name="base_architecture", node=hc.ArchitectureConfig)
    cs.store(group="architecture/resnet50", name="base_resnet50", node=hc.ResNet50Config)

    # Register training configs
    cs.store(group="training", name="base_training", node=hc.TrainingConfig)
    cs.store(group="optimizer", name="base_optimizer", node=hc.OptimizerConfig)
    cs.store(group="loss", name="base_loss", node=hc.LossConfig)


# Auto-register configs when module is imported
register_configs()
