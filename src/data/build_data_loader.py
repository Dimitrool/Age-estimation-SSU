import torch

from typing import Literal

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.ImagePairDataset import ImagePairDataset
from src.data.data_augmentation import build_transforms
from src.constants import SPLIT_TO_PATH
from src.utils.filesystem_utils import read_input


def build_dataloader(cfg: DictConfig, split: Literal["training", "validation", "testing"]):
    data = read_input(str(SPLIT_TO_PATH[split]))
    preprocess = build_transforms(cfg.data.augmentations[split], cfg.data.image_size)
    dataset = ImagePairDataset(data, preprocess)

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        shuffle=(split == "training"),
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )


def collate_fn(batch):
    """
    A custom 'collate' function to combine individual samples into a batch.

    The DataLoader fetches individual samples from the ImagePairDataset as tuples.
    This function's job is to take a list of these tuples (the 'batch') and
    transform it into a single tuple of batched tensors.

    Args:
        batch (list):
            A list of samples, where each sample is the tuple returned by
            `ImagePairDataset.__getitem__`. For example:
            [(tensor1_sample1, tensor2_sample1, age1_sample1),
             (tensor1_sample2, tensor2_sample2, age1_sample2), ...]

    Returns:
        tuple: A tuple of tensors, where each tensor represents the entire batch:
            - tensors1_stacked (torch.Tensor): A tensor of shape (N, C, H, W)
              for the first images in the batch.
            - tensors2_stacked (torch.Tensor): A tensor of shape (N, C, H, W)
              for the second images in the batch.
            - true_age1s_tensor (torch.Tensor): A tensor of shape (N,) for the
              ages of the first faces.
    """
    # Unzip the batch into separate lists
    tensors1, tensors2, true_age1s = zip(*batch)
    
    # Stack the tensors and convert ages to tensors
    tensors1_stacked = torch.stack(tensors1)
    tensors2_stacked = torch.stack(tensors2)
    true_age1s_tensor = torch.tensor(true_age1s, dtype=torch.float32)
        
    return tensors1_stacked, tensors2_stacked, true_age1s_tensor
