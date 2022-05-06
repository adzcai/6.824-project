import gzip
import hashlib
import os
import numpy as np
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.net(x)
        output = F.log_softmax(x, dim=1)
        return output


def fetch_np_array(url: str, path: str) -> np.ndarray:
    """
    This function retrieves the data from the given url and caches the data locally in the path so that we do not
    need to repeatedly download the data every time we call this function.
    Args:
        url: link from which to retrieve data
        path: path on local desktop to save file
    Returns:
        Numpy array that is fetched from the given url or retrieved from the cache.
    """
    path = os.path.join(os.getcwd(), path)
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(fp, "wb+") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def get_mnist_data(partition=-1, split="test"):
    data_cache = "data\\"
    if split == "test":
        X_test = fetch_np_array(
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", data_cache
        )[0x10:].reshape((-1, 28, 28))
        y_test = fetch_np_array("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", data_cache)[8:]
        return X_test, y_test
    elif split == "train":
        assert 0 <= partition < 10, f"Partition {partition} is invalid"
        X_train = fetch_np_array(
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", data_cache
        )[0x10:].reshape((-1, 28, 28))
        y_train = fetch_np_array("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", data_cache)[8:]
        splits = np.split(np.arange(X_train.shape[0]), 10)
        return X_train[splits[partition]], y_train[splits[partition]]
    else:
        raise ValueError(f"Invalid split type {split}")
