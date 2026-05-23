import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from sklearn.model_selection import StratifiedKFold
from .preprocessing import (
    parse_csv, sliding_window, compute_normalization_params, normalize,
)
from src.utils.helpers import CLASS_TO_IDX, set_seed


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


def load_all_data(
    train_dir: str,
    test_dir: str,
    window_size: int,
    stride: int,
    norm_method: str = "global",
) -> tuple:
    """Load ALL CSV files.

    norm_method:
      "global"   — return raw windows (caller does one global normalization)
      "per_file" — normalize each file independently using its own stats

    Returns (X, y) where X is (N, L, C).
    """
    train_dir = Path(train_dir)

    all_windows = []
    all_labels = []

    def _load_one_file(file_path: Path, class_idx: int):
        sensor_data = parse_csv(str(file_path))
        windows = sliding_window(sensor_data, window_size, stride)
        if norm_method == "per_file":
            params = compute_normalization_params(windows)
            windows = normalize(windows, params)
        all_windows.append(windows)
        all_labels.append(np.full(len(windows), class_idx, dtype=np.int64))

    # load from train_dir (organized by class subdirectories)
    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue
        for csv_file in sorted(class_dir.glob("*.csv")):
            _load_one_file(csv_file, class_idx)

    # load from test_dir (flat, infer class from filename)
    test_dir = Path(test_dir)
    if test_dir.exists():
        for csv_file in sorted(test_dir.glob("*.csv")):
            fname = csv_file.stem
            matched = None
            for cls_name in CLASS_TO_IDX:
                if cls_name in fname:
                    matched = cls_name
                    break
            if matched is None:
                continue
            _load_one_file(csv_file, CLASS_TO_IDX[matched])

    if len(all_windows) == 0:
        raise ValueError("No data found")

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def create_kfold_dataloaders(
    config,
    fold: int,
    transform: Optional[Callable] = None,
):
    """Create DataLoaders for a specific fold of StratifiedKFold."""
    set_seed(config.get("data", "random_seed", default=42))

    norm_method = config.get("data", "normalization", default="zscore")

    X, y = load_all_data(
        train_dir=config.get("data", "train_dir"),
        test_dir=config.get("data", "test_dir"),
        window_size=config.get("data", "window_size"),
        stride=config.get("data", "stride"),
        norm_method=norm_method,
    )

    n_splits = config.get("kfold", "n_splits", default=5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.get("data", "random_seed", default=42))
    folds = list(skf.split(X, y))
    train_idx, val_idx = folds[fold]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    if norm_method == "per_file":
        # already normalized per-file; no additional normalization
        norm_params = None
    else:
        # global per-fold normalization on training fold only
        norm_params = compute_normalization_params(X_train)
        X_train = normalize(X_train, norm_params)
        X_val = normalize(X_val, norm_params)

    # transpose to (N, C, L)
    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    batch_size = config.get("training", "batch_size", default=128)
    num_workers = config.get("data", "num_workers", default=4)

    train_dataset = MattressDataset(X_train, y_train, transform=transform)
    val_dataset = MattressDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, (train_idx, val_idx), norm_params
