import torch

from typing import Literal

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.ImagePairDataset import ImagePairDataset
from src.data.Samplers import IdentityBalancedSampler
from src.data.data_augmentation import build_transforms
from src.data.read_data import read_input, read_self_augmented_input
from src.constants import SPLIT_TO_PATH
from src.data.ImagePairDataset import collate_fn


def build_data_loader(cfg: DictConfig, split: Literal["training", "validation", "testing"]) -> DataLoader:
    shuffle = (split == "training" or cfg.training.use_full_dataset)

    data_path = str(SPLIT_TO_PATH[split])
    if cfg.data.use_self_augmentation and shuffle:
        data = read_self_augmented_input(data_path)
    else:
        data = read_input(data_path)

    preprocess = build_transforms(cfg.data.augmentations[split], cfg.data.image_size)
    dataset = ImagePairDataset(data, preprocess)

    ppl_per_batch = cfg.data.pairs_per_person_in_batch
    if shuffle and ppl_per_batch is not None and 0 < ppl_per_batch < cfg.data.batch_size and cfg.data.batch_size % ppl_per_batch == 0:
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
