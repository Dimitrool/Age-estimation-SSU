import torch

from typing import Literal

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.ImagePairDataset import ImagePairDataset
from src.data.Samplers import IdentityBalancedSampler
from src.data.data_augmentation import build_transforms
from src.data.read_data import read_input, read_self_augmented_input
from src.constants import SPLIT_TO_PATH


def build_data_loader(cfg: DictConfig, split: Literal["training", "validation", "testing"]) -> DataLoader:
    data_path = str(SPLIT_TO_PATH[split])
    if cfg.data.use_self_augmentation and split == "training":
        data = read_self_augmented_input(data_path)
    else:
        data = read_input(data_path)

    preprocess = build_transforms(cfg.data.augmentations[split], cfg.data.image_size)
    dataset = ImagePairDataset(data, preprocess)

    shuffle = (split == "training")
    ppl_per_batch = cfg.data.pairs_per_person_in_batch
    if split == "training" and ppl_per_batch is not None and 0 < ppl_per_batch < cfg.data.batch_size and cfg.data.batch_size % ppl_per_batch == 0:
        sampler = IdentityBalancedSampler(
            dataset, 
            batch_size=cfg.data.batch_size, 
            samples_per_person=ppl_per_batch
        )
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
