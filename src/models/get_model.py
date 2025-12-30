import os
import torch
import torch.nn as nn
from pathlib import Path
import torchvision.models as models
from omegaconf import DictConfig
from typing import Literal
import re

import src.models.wrappers as wrp
import src.models.architectures as arch # uses arch btw
from src.constants import CHECKPOINTS_FOLDER_NAME, MAX_AGE


BASELINE_WRAPPER_REGISTRY = {
    "resnet50": wrp.ResNet50BaselineWrapper
}


def load_model_weights(model: nn.Module, weights_source: Path, device: torch.device):
    """
    Loads weights into the model from a file path.
    """
    if os.path.isfile(weights_source):
        print(f"Loading weights from: {weights_source}")
        # Map location ensures weights are loaded to the correct device
        state_dict = torch.load(weights_source, map_location=device)
        
        # Handle cases where the state_dict might be nested (e.g. {"state_dict": ...})
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        # Load weights, ignoring mismatch if necessary (e.g. different head)
        # strict=False is useful if we are loading a backbone into a full model container
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Weights file not found at {weights_source}. Using random init.")

def get_backbone(backbone_cfg: DictConfig, device: torch.device, use_cfg_weights: bool = True) -> nn.Module:
    """
    Instantiates the backbone model (e.g., ResNet50).
    """
    model_name = backbone_cfg.name
    # Specific config for the chosen model (e.g., cfg.model.backbone.resnet50)
    model_specific_cfg = getattr(backbone_cfg, model_name)
    
    source = model_specific_cfg.source
    pretrained = model_specific_cfg.pretrained
    weights_source = model_specific_cfg.weights_source

    print(f"Initializing Backbone: {model_name} from {source}")

    if source == "torchvision.models":
        # Check if we use official weights or random init
        weights = None
        if pretrained and weights_source == "torch" and use_cfg_weights:
            # Use default (best available) official weights
            weights = "DEFAULT" 
        
        # Instantiate from torchvision
        if hasattr(models, model_name):
            backbone = getattr(models, model_name)(num_classes = MAX_AGE, weights=weights)
        else:
            raise ValueError(f"Model {model_name} not found in torchvision.models")

        # If pretrained is True but source is NOT "torch", we load custom weights file
        if pretrained and weights_source != "torch" and use_cfg_weights:
            load_model_weights(backbone, Path(weights_source), device)

    elif source == "local":
        # Support for custom architectures defined in src.models.architectures
        if hasattr(arch, model_name):
            backbone = getattr(arch, model_name)(num_classes = MAX_AGE)
            if pretrained and weights_source and use_cfg_weights:
                load_model_weights(backbone, Path(weights_source), device)
        else:
            raise ValueError(f"Custom model {model_name} not found in src.models.architectures")
    else:
        raise ValueError(f"Unknown model source: {source}")

    return backbone

def wrap_with_postprocessing(cfg: DictConfig, backbone: nn.Module, device) -> wrp.BaseBackboneWrapper:
    if cfg.model.backbone.name not in BASELINE_WRAPPER_REGISTRY:
        raise ValueError(f"Unknown model: {cfg.model.backbone.name}")

    baseline_wrapper = BASELINE_WRAPPER_REGISTRY[cfg.model.backbone.name](backbone, cfg.data.image_size)
    baseline_wrapper.to(device)
    return baseline_wrapper

def build_head(head_cfg: DictConfig, in_channels: int, out_channels: int) -> nn.Module:
    """
    Constructs an MLP head using the params from the config.
    """
    dropout_p = getattr(head_cfg, "dropout_p", [])
    
    return arch.Head(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=head_cfg.hidden_channels,
        activation=head_cfg.activation,
        dropout_p=dropout_p
    )

def wrap_baseline_wrapper(wrapper_cfg: DictConfig, baseline_wrapper: wrp.BaseBackboneWrapper) -> nn.Module:
    wrapper_name = wrapper_cfg.name
    wrapper_cfg = getattr(wrapper_cfg, wrapper_name)
    
    print(f"Wrapping model with strategy: {wrapper_name}")

    if wrapper_name == "const_offset":
        model = wrp.ConstOffsetCorrectionWrapper(
            backbone=baseline_wrapper, 
            factor=wrapper_cfg.factor
        )

    elif wrapper_name == "learnable_offset":
        # Learnable Offset maps 1D Error -> 1D Correction
        head = build_head(wrapper_cfg.head, in_channels=baseline_wrapper.feature_dim, out_channels=1)
        model = wrp.LearnableOffsetCorrectionWrapper(
            base_model=baseline_wrapper,
            error_mapper=head
        )

    elif wrapper_name == "delta_regression":
        head = build_head(wrapper_cfg.head, in_channels=baseline_wrapper.feature_dim, out_channels=1)
        model = wrp.DeltaRegressionWrapper(
            backbone=baseline_wrapper,
            delta_head=head
        )

    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_name}")
    
    return model

def get_model(cfg: DictConfig, device: torch.device, load_model_weights = True) -> nn.Module:
    """
    Main entry point to build the full model pipeline.
    """
    # 1. Instantiate the Backbone (e.g., ResNet50)
    backbone = get_backbone(cfg.model.backbone, device, load_model_weights)

    # 2. Wrap it in the Baseline Wrapper (Features + Logits)
    # This wrapper handles the separation of features and classification head
    baseline_wrapper = wrap_with_postprocessing(cfg, backbone, device)

    # 3. Apply the Logic Wrapper (Strategy)
    model = wrap_baseline_wrapper(cfg.model.wrapper, baseline_wrapper)

    model.to(device)
    return model

def get_checkpoint_path(results_path: Path, mode: Literal["best", "latest"]) -> Path:
    """Finds the checkpoint with the highest epoch number (epoch_XXXX.pth)."""
    experiment_dir = results_path / CHECKPOINTS_FOLDER_NAME

    if mode == "best":
        return experiment_dir / "best_checkpoint.pth"

    # Look for files matching epoch_*.pth
    files = list(experiment_dir.glob("epoch_*.pth"))

    if not files:
        raise FileNotFoundError(f"Checkpointa were not found at {experiment_dir}")

    def extract_epoch_num(file_path):
        # Extract integer from filename (e.g., 0010 from epoch_0010.pth)
        match = re.search(r'(\d+)', file_path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=extract_epoch_num)

def get_pretrained_model(cfg: DictConfig, results_path: Path, device):
    use_backbone_weights = (".pth" in str(results_path))

    model = get_model(cfg, device, use_backbone_weights)

    if not use_backbone_weights:
        path_to_weights = get_checkpoint_path(results_path, "best")
        load_model_weights(model, path_to_weights, device)

    return model

