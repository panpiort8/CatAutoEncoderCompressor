from pathlib import Path
from typing import Tuple

import numpy as np
import torch as T
from PIL import Image
from torch.utils.data import Dataset


def load_patches(file_name: str):
    img = np.array(Image.open(file_name))

    pad = ((24, 24), (0, 0), (0, 0))

    if img.shape == (720, 1280, 3):
        img = np.pad(img, pad, mode="edge")
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = T.from_numpy(img).float()

    patches = np.reshape(img, (3, 6, 128, 10, 128))
    patches = np.transpose(patches, (0, 1, 3, 2, 4))

    return img, patches


class ImageFolder720p(Dataset):
    """
    Image shape is (720, 1280, 3) --> (768, 1280, 3) --> 6x10 128x128 patches
    """

    def __init__(self, root: str):
        self.files = sorted(Path(root).iterdir())

    def __getitem__(self, index: int) -> Tuple[T.Tensor, np.ndarray, str]:
        path = str(self.files[index % len(self.files)])
        img, patches = load_patches(file_name=path)
        return img, patches, path

    def __len__(self):
        return len(self.files)
