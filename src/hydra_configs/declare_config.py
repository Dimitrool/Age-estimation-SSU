from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.constants import HYDRA_OUTPUT


# --- Augmentation Sub-Components ---
@dataclass
class AugmentationItem:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AugmentationsConfig:
    training: List[AugmentationItem] = field(default_factory=list)
    validation: List[AugmentationItem] = field(default_factory=list)
    testing: List[AugmentationItem] = field(default_factory=list)

# --- Data Config ---
@dataclass
class DataConfig:
    use_full_dataset: bool = False
    image_size: int = 256
    batch_size: int = 32
    num_workers: int = 16
    use_self_augmentation: bool = True
    pairs_per_person_in_batch: Optional[int] = 4
    augmentations: AugmentationsConfig = field(default_factory=AugmentationsConfig)

# --- Model & Wrapper Configs ---
@dataclass
class ResNet50Config:
    source: str = "torchvision.models"
    pretrained: bool = True
    weights_source: str = "weights/age_resnet50.pth"


@dataclass
class HeadFunctionalConfig:
    hidden_channels: List[int] = [16]
    activation: str = "relu"


@dataclass
class HeadNetConfig:
    hidden_channels: List[int] = [512]
    activation: str = "relu"
    dropout_p: List[float] = [0.4]


@dataclass
class ConstOffsetCorrectionWrapperConfig:
    factor: float = 0.5


@dataclass
class LearnableOffsetCorrectionWrapperConfig:
    head: HeadFunctionalConfig = field(default_factory=HeadFunctionalConfig)


@dataclass
class DeltaRegressionWrapperConfig:
    head: HeadNetConfig = field(default_factory=HeadNetConfig)


@dataclass
class WrapperConfig:
    """
    Configuration for the logic wrapping the backbone.
    Types: 'baseline', 'const_offset', 'learnable_offset', 'delta_regression'
    """
    name: str = "const_offset"
    const_offset: ConstOffsetCorrectionWrapperConfig = field(default_factory=ConstOffsetCorrectionWrapperConfig)
    learnable_offset: LearnableOffsetCorrectionWrapperConfig = field(default_factory=LearnableOffsetCorrectionWrapperConfig)
    delta_regression: DeltaRegressionWrapperConfig = field(default_factory=DeltaRegressionWrapperConfig)

@dataclass
class BackboneConfig:
    name: str = "resnet50"
    resnet50: ResNet50Config = field(default_factory=ResNet50Config)

@dataclass
class ModelConfig:
    """
    Renamed from ArchitectureConfig.
    Contains the backbone (ResNet) and the wrapper strategy.
    """
    backbone: ResNet50Config = field(default_factory=ResNet50Config)
    wrapper: WrapperConfig = field(default_factory=WrapperConfig)


# --- Optimizer & Training ---
@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.99, 0.9995])
    weight_decay: float = 1e-5
    scheduler_gamma: float = 0.995

@dataclass
class TrainingConfig:
    num_epochs: int = 100
    patience: int = 15
    checkpoint_interval: int = 5
    accumulation_steps: int = 1

@dataclass
class LossConfig:
    type: str = "l1"

@dataclass
class LogConfig:
    tensorboard: bool = True

# --- Top Level Configuration ---
@dataclass
class BaseConfig:
    experiment_name: str = "resnet50_delta_reg"
    data: DataConfig = field(default_factory=DataConfig)
    
    # Renamed field
    model: ModelConfig = field(default_factory=ModelConfig)
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    hydra: Any = field(default_factory=lambda: {
        "run": {
            "dir": f"{HYDRA_OUTPUT}/${{experiment_name}}/${{now:%Y-%m-%d_%H-%M}}"
        },
        "sweep": {
            "dir": f"{HYDRA_OUTPUT}/${{experiment_name}}/${{now:%Y-%m-%d_%H-%M}}",
            "subdir": "${hydra.job.override_dirname}"
        },
        "job_logging": {
            "handlers": {
                "file": {
                    "filename": "training.log"
                }
            }
        }
    })
