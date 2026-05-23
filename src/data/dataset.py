import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Optional, Callable
from .preprocessing import load_all_train_data, load_test_data, compute_normalization_params, normalize
from src.utils.helpers import set_seed


class MattressDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform: Optional[Callable] = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]


def create_dataloaders(
    config,
    transform: Optional[Callable] = None,
) -> tuple:
    set_seed(config.get("data", "random_seed", default=42))

    X, y, norm_params = load_all_train_data(
        train_dir=config.get("data", "train_dir"),
        window_size=config.get("data", "window_size"),
        stride=config.get("data", "stride"),
        norm_method=config.get("data", "normalization"),
    )

    val_ratio = config.get("data", "val_ratio", default=0.1)
    test_ratio = config.get("data", "test_ratio", default=0.1)

    n_total = len(y)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    full_dataset = MattressDataset(X, y, transform=transform)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test],
    )

    # test_dataset from train_data (no transform)
    val_dataset.dataset.transform = None
    test_dataset.dataset.transform = None

    batch_size = config.get("training", "batch_size", default=128)
    num_workers = config.get("data", "num_workers", default=4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # independent test data
    X_test, y_test = load_test_data(
        test_dir=config.get("data", "test_dir"),
        window_size=config.get("data", "window_size"),
        stride=config.get("data", "stride"),
        norm_params=norm_params,
        norm_method=config.get("data", "normalization"),
    )
    test_extra_dataset = MattressDataset(X_test, y_test)
    test_extra_loader = DataLoader(
        test_extra_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_extra_loader
