from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.constants import HYDRA_OUTPUT


# --- Augmentation Sub-Components ---
@dataclass
class AugmentationItem:
    """Represents a single augmentation step (e.g., RandomCrop)"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AugmentationsConfig:
    """Defines augmentations for each data split"""
    training: List[AugmentationItem] = field(default_factory=list)
    validation: List[AugmentationItem] = field(default_factory=list)
    testing: List[AugmentationItem] = field(default_factory=list)

# --- Main Section Configs ---
@dataclass
class DataConfig:
    image_size: int = 256
    batch_size: int = 32
    num_workers: int = 16
    use_self_augmentation: bool = True
    pairs_per_person_in_batch: int | None = 4
    augmentations: AugmentationsConfig = field(default_factory=AugmentationsConfig)


@dataclass
class ResNet50Config:
    source: str = "torchvision.models"
    pretrained: bool = True
    weights_source: str = "age_resnet50.pth"


@dataclass
class ArchitectureConfig:
    name: str = "resnet50"
    resnet50: ResNet50Config = field(default_factory=ResNet50Config)


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

@dataclass
class LossConfig:
    type: str = "l1"

# --- Top Level Configuration ---
@dataclass
class BaseConfig:
    """
    The main configuration schema.
    """
    experiment_name: str = "resnet50_baseline"
    data: DataConfig = field(default_factory=DataConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
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



