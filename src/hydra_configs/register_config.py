import src.hydra_configs.declare_config as hc
from hydra.core.config_store import ConfigStore


def register_configs() -> None:
    cs = ConfigStore.instance()

    cs.store(name="baseline", node=hc.BaseConfig)

    # Data
    cs.store(group="data", name="base_data", node=hc.DataConfig)
    cs.store(group="data/augmentations", name="base_augmentations", node=hc.AugmentationsConfig)

    # Model (Previously Architecture)
    cs.store(group="model", name="base_model", node=hc.ModelConfig)
    cs.store(group="model/backbone", name="base_backbone", node=hc.BackboneConfig)
    cs.store(group="model/wrapper", name="base_wrapper", node=hc.WrapperConfig)

    cs.store(group="model/backbone/resnet50", name="base_resnet50", node=hc.ResNet50Config)

    cs.store(group="model/wrapper/const_offset", name="base_const_offset", node=hc.ConstOffsetCorrectionWrapperConfig)
    cs.store(group="model/wrapper/learnable_offset", name="base_learnable_offset", node=hc.LearnableOffsetCorrectionWrapperConfig)
    cs.store(group="model/wrapper/delta_regression", name="base_delta_regression", node=hc.DeltaRegressionWrapperConfig)

    # Training
    cs.store(group="training", name="base_training", node=hc.TrainingConfig)
    cs.store(group="optimizer", name="base_optimizer", node=hc.OptimizerConfig)
    cs.store(group="loss", name="base_loss", node=hc.LossConfig)
    cs.store(group="log", name="base_log", node=hc.LogConfig)

register_configs()
