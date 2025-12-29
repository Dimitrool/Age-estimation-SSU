#!/usr/bin/env python3
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImagePairDataset(Dataset):
    """
    A PyTorch Dataset to load and preprocess pairs of face images.

    This class is designed to work with a DataLoader. It takes a dictionary of
    data containing file paths and ages, and when an item is requested (e.g.,
    by the DataLoader), it loads the corresponding pair of images, applies
    the necessary transformations, and returns them as tensors along with the
    known age of the first person.

    Args:
        data (dict):
            A dictionary containing the dataset information. Expected keys are:
            'face1_path': A list of file paths for the first face images.
            'age1': A list of corresponding ages for the first face images.
            'face2_path': A list of file paths for the second face images.
        preprocess (transforms.Compose):
            A sequence of torchvision transforms to be applied to each image
            after it is loaded.
    """
    def __init__(self, data: dict, preprocess: transforms.Compose):
        self.face1_paths = data["face1_path"]
        self.true_age1s = data["age1"]
        self.face2_paths = data["face2_path"]
        self.true_age2s = data["age2"]
        self.person_ids = data["person_id"]
        self.preprocess = preprocess

    def __len__(self) -> int:
        """
        Returns the total number of samples (image pairs) in the dataset.
        """
        return len(self.face1_paths)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves and preprocesses a single data sample from the dataset.

        This method is called by the DataLoader for each index in a batch. It
        opens the two images corresponding to the given index, converts them to
        RGB, applies the preprocessing transforms, and returns them as tensors.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - tensor1 (torch.Tensor): The preprocessed tensor for the first image.
                - tensor2 (torch.Tensor): The preprocessed tensor for the second image.
                - true_age1 (float): The known age of the person in the first image.
        """
        path1 = self.face1_paths[index]
        path2 = self.face2_paths[index]
        
        image1 = Image.open(path1).convert("RGB")
        tensor1 = self.preprocess(image1)

        image2 = Image.open(path2).convert("RGB")
        tensor2 = self.preprocess(image2)

        true_age1 = self.true_age1s[index]
        true_age2 = self.true_age1s[index]
        
        return tensor1, tensor2, true_age1, true_age2
