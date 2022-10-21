"""
Modified from https://mattpetersen.github.io/load-mnist-with-numpy
"""
import gzip

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from typing import Union, Optional, Callable, Generator


def standard_transform(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255


def standard_target_transform(lbls: np.ndarray) -> np.ndarray:
    n_rows = len(lbls)
    n_cols = lbls.max() + 1
    onehot = np.zeros((n_rows, n_cols), dtype=np.uint8)
    onehot[np.arange(n_rows), lbls] = 1
    return onehot


class MNIST:
    URL = "http://yann.lecun.com/exdb/mnist/"
    FILES = {
        True: {  # True -> train
            "images": "train-images-idx3-ubyte.gz",
            "labels": "train-labels-idx1-ubyte.gz",
        },
        False: {  # False -> not train = test
            "images": "t10k-images-idx3-ubyte.gz",
            "labels": "t10k-labels-idx1-ubyte.gz",
        },
    }

    def __init__(
        self,
        data_root: Union[str, Path],
        train: bool,
        download: bool = False,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        target_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.data_root = Path(data_root)
        self.data_path = self.data_root / "mnist"  # sub-folder for mnist

        # Create path if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.train = train
        self.download = download

        self.transform = transform or standard_transform
        self.target_transform = target_transform or standard_target_transform

        self.images = None
        self.labels = None

        self._load_data()

    def _load_data(self) -> None:
        self._download_if_needed()
        self._load_images()
        self._load_labels()

        assert self.images is not None
        assert self.labels is not None
        assert len(self.images) == len(self.labels)

    def _download_if_needed(self):
        # Download any missing files
        for file in self.FILES[self.train].values():
            path = self.data_path / file
            if not path.exists():
                urlretrieve(self.URL + file, path)
                print(f"Downloaded {file} to {path}")

    def _load_images(self):
        path = self.data_path / self.FILES[self.train]["images"]
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            images = np.frombuffer(f.read(), "B", offset=16)

        images = images.reshape(-1, 28, 28)
        if self.transform is not None:
            images = self.transform(images)

        self.images = images

    def _load_labels(self):
        path = self.data_path / self.FILES[self.train]["labels"]
        with gzip.open(path) as f:
            # First 8 bytes are magic nums
            labels = np.frombuffer(f.read(), "B", offset=8)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        self.labels = labels

    def __getitem__(self, item):
        img = self.images[item]
        lbl = self.labels[item]
        return img, lbl

    def __len__(self):
        return len(self.images)

    def as_batched(self, batch_size: int, drop_last: bool = False) -> Generator:
        """
        Returns a generator, which returns batched images and labels of given
        batch_size.

        Parameters
        ----------
        batch_size : int
            The batch size to use.
        drop_last : bool
            Whether to drop the last batch if its size is less than batch_size.
        """
        i = 0
        size = len(self)
        while i < size:
            lbls = self.labels[i:i+batch_size]

            if drop_last and len(lbls) < batch_size:
                break

            imgs = self.images[i:i+batch_size]
            yield imgs, lbls

            i += batch_size
