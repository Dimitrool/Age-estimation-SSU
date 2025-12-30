import torch
import torch.nn as nn
from pathlib import Path
import torchvision.models as models
from typing import Literal, Dict, Any
import re
import os

import src.models.wrappers as wrp
import src.models.architectures as arch # uses arch btw
from src.constants import CHECKPOINTS_FOLDER_NAME, MAX_AGE

# Registry for baseline wrappers
BASELINE_WRAPPER_REGISTRY = {
    "resnet50": wrp.ResNet50BaselineWrapper
}

def load_model_weights(model: nn.Module, weights_source: Path, device: torch.device):
    """
    Loads weights into the model from a file path.
    """
    prefix = os.environ.get('UPLOAD_DIR')
    if prefix:
        weights_source = prefix / weights_source

    print(f"Loading weights from: {weights_source}")
    # Map location ensures weights are loaded to the correct device
    state_dict = torch.load(weights_source, map_location=device)
    
    # Handle cases where the state_dict might be nested (e.g. {"state_dict": ...})
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
        
    # Load weights, ignoring mismatch if necessary (e.g. different head)
    model.load_state_dict(state_dict, strict=False)

def get_backbone(backbone_cfg: Dict[str, Any], device: torch.device, use_backbone_weights: bool = True) -> nn.Module:
    """
    Instantiates the backbone model (e.g., ResNet50).
    Args:
        backbone_cfg: Dictionary containing backbone config (e.g. {'name': 'resnet50', 'resnet50': {...}})
    """
    model_name = backbone_cfg['name']
    
    # Access specific config: cfg['resnet50']
    model_specific_cfg = backbone_cfg[model_name]
    
    source = model_specific_cfg['source']
    pretrained = model_specific_cfg['pretrained']
    weights_source = model_specific_cfg['weights_source']

    print(f"Initializing Backbone: {model_name} from {source}")

    if source == "torchvision.models":
        weights = None
        if pretrained and weights_source == "torch" and use_backbone_weights:
            weights = "DEFAULT" 
        
        if hasattr(models, model_name):
            backbone = getattr(models, model_name)(num_classes=MAX_AGE, weights=weights)
        else:
            raise ValueError(f"Model {model_name} not found in torchvision.models")

        if pretrained and weights_source != "torch" and use_backbone_weights:
            load_model_weights(backbone, Path(weights_source), device)

    elif source == "local":
        if hasattr(arch, model_name):
            backbone = getattr(arch, model_name)(num_classes=MAX_AGE)
            if pretrained and weights_source and use_backbone_weights:
                load_model_weights(backbone, Path(weights_source), device)
        else:
            raise ValueError(f"Custom model {model_name} not found in src.models.architectures")
    else:
        raise ValueError(f"Unknown model source: {source}")

    return backbone

def wrap_with_postprocessing(cfg: Dict[str, Any], backbone: nn.Module, device) -> wrp.BaseBackboneWrapper:
    # Access: cfg['model']['backbone']['name']
    backbone_name = cfg['model']['backbone']['name']
    
    if backbone_name not in BASELINE_WRAPPER_REGISTRY:
        raise ValueError(f"Unknown model: {backbone_name}")

    # Access: cfg['data']['image_size']
    img_size = cfg['data']['image_size']
    
    baseline_wrapper = BASELINE_WRAPPER_REGISTRY[backbone_name](backbone, img_size)
    baseline_wrapper.to(device)
    return baseline_wrapper

def build_head(head_cfg: Dict[str, Any], in_channels: int, out_channels: int) -> nn.Module:
    """
    Constructs an MLP head using the params from the config dict.
    """
    # Use .get() for optional parameters
    dropout_p = head_cfg.get("dropout_p", [])
    
    return arch.Head(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=head_cfg['hidden_channels'],
        activation=head_cfg['activation'],
        dropout_p=dropout_p
    )

def wrap_baseline_wrapper(wrapper_cfg: Dict[str, Any], baseline_wrapper: wrp.BaseBackboneWrapper) -> nn.Module:
    wrapper_name = wrapper_cfg['name']
    # Access specific wrapper config: cfg['delta_regression']
    specific_cfg = wrapper_cfg[wrapper_name]
    
    print(f"Wrapping model with strategy: {wrapper_name}")

    if wrapper_name == "const_offset":
        model = wrp.ConstOffsetCorrectionWrapper(
            backbone=baseline_wrapper, 
            factor=specific_cfg['factor']
        )

    elif wrapper_name == "learnable_offset":
        head = build_head(specific_cfg['head'], in_channels=baseline_wrapper.feature_dim, out_channels=1)
        model = wrp.LearnableOffsetCorrectionWrapper(
            base_model=baseline_wrapper,
            error_mapper=head
        )

    elif wrapper_name == "delta_regression":
        head = build_head(specific_cfg['head'], in_channels=baseline_wrapper.feature_dim, out_channels=1)
        model = wrp.DeltaRegressionWrapper(
            backbone=baseline_wrapper,
            delta_head=head
        )

    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_name}")
    
    return model

def get_model(cfg: Dict[str, Any], device: torch.device, use_backbone_weights = True) -> nn.Module:
    """
    Main entry point to build the full model pipeline.
    Expected cfg is a standard Python dictionary.
    """
    # 1. Instantiate the Backbone
    backbone = get_backbone(cfg['model']['backbone'], device, use_backbone_weights)

    # 2. Wrap it in the Baseline Wrapper
    baseline_wrapper = wrap_with_postprocessing(cfg, backbone, device)

    # 3. Apply the Logic Wrapper
    model = wrap_baseline_wrapper(cfg['model']['wrapper'], baseline_wrapper)

    model.to(device)
    return model

def get_checkpoint_path(results_path: Path, mode: Literal["best", "latest"]) -> Path:
    """Finds the checkpoint with the highest epoch number (epoch_XXXX.pth)."""
    experiment_dir = results_path / CHECKPOINTS_FOLDER_NAME

    if mode == "best":
        return experiment_dir / "best_checkpoint.pth"

    files = list(experiment_dir.glob("epoch_*.pth"))

    if not files:
        raise FileNotFoundError(f"Checkpoints were not found at {experiment_dir}")

    def extract_epoch_num(file_path):
        match = re.search(r'(\d+)', file_path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=extract_epoch_num)

def get_pretrained_model(cfg: Dict[str, Any], results_path: Path, device):
    use_backbone_weights = ("age_resnet50.pth" in str(results_path))

    model = get_model(cfg, device, use_backbone_weights)

    if not use_backbone_weights:
        path_to_weights = get_checkpoint_path(results_path, "best")
        load_model_weights(model, path_to_weights, device)

    return model
