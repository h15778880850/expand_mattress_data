import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def parse_csv(file_path: str) -> np.ndarray:
    df = pd.read_csv(file_path, header=None)
    # The first column is a timestamp (either relative seconds or absolute datetime).
    # We only need the 7 sensor columns (columns 1-7).
    data = df.iloc[:, 1:].values.astype(np.float32)
    return data


def sliding_window(
    data: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    samples = []
    start = 0
    while start + window_size <= len(data):
        samples.append(data[start:start + window_size])
        start += stride
    if len(samples) == 0:
        if len(data) >= window_size:
            samples.append(data[:window_size])
    return np.array(samples, dtype=np.float32)


def compute_normalization_params(
    samples: np.ndarray,
    method: str = "zscore",
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "zscore":
        mean = np.mean(samples, axis=(0, 1))
        std = np.std(samples, axis=(0, 1))
        std = np.where(std < 1e-8, 1e-8, std)
        return mean, std
    elif method == "minmax":
        min_val = np.min(samples, axis=(0, 1))
        max_val = np.max(samples, axis=(0, 1))
        return min_val, max_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize(
    samples: np.ndarray,
    params: Tuple[np.ndarray, np.ndarray],
    method: str = "zscore",
) -> np.ndarray:
    if method == "zscore":
        mean, std = params
        return (samples - mean) / std
    elif method == "minmax":
        min_val, max_val = params
        range_val = max_val - min_val
        range_val = np.where(range_val < 1e-8, 1e-8, range_val)
        return (samples - min_val) / range_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_csv(
    file_path: str,
    window_size: int,
    stride: int,
    norm_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    norm_method: str = "zscore",
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    sensor_data = parse_csv(file_path)
    windows = sliding_window(sensor_data, window_size, stride)
    if norm_params is None:
        norm_params = compute_normalization_params(windows, method=norm_method)
    windows = normalize(windows, norm_params, method=norm_method)
    # transpose to (N, C, L)
    windows = windows.transpose(0, 2, 1)
    return windows, norm_params


def load_all_train_data(
    train_dir: str,
    window_size: int,
    stride: int,
    norm_method: str = "zscore",
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    from src.utils.helpers import CLASS_TO_IDX
    train_dir = Path(train_dir)
    all_windows = []
    all_labels = []
    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue
        for csv_file in sorted(class_dir.glob("*.csv")):
            sensor_data = parse_csv(str(csv_file))
            windows = sliding_window(sensor_data, window_size, stride)
            all_windows.append(windows)
            all_labels.append(np.full(len(windows), class_idx, dtype=np.int64))
    if len(all_windows) == 0:
        raise ValueError("No training data found")
    X = np.concatenate(all_windows, axis=0)  # (N, L, C)
    y = np.concatenate(all_labels, axis=0)
    norm_params = compute_normalization_params(X, method=norm_method)
    X = normalize(X, norm_params, method=norm_method)
    X = X.transpose(0, 2, 1)  # (N, C, L)
    return X, y, norm_params


def load_test_data(
    test_dir: str,
    window_size: int,
    stride: int,
    norm_params: Tuple[np.ndarray, np.ndarray],
    norm_method: str = "zscore",
) -> Tuple[np.ndarray, np.ndarray]:
    from src.utils.helpers import CLASS_TO_IDX
    test_dir = Path(test_dir)
    all_windows = []
    all_labels = []
    for csv_file in sorted(test_dir.glob("*.csv")):
        # infer class label from filename (e.g. 01_supine.csv -> supine)
        fname = csv_file.stem
        matched_class = None
        for class_name in CLASS_TO_IDX:
            if class_name in fname:
                matched_class = class_name
                break
        if matched_class is None:
            print(f"Warning: cannot infer class from {fname}, skipping")
            continue
        class_idx = CLASS_TO_IDX[matched_class]
        windows, _ = preprocess_csv(
            str(csv_file), window_size, stride,
            norm_params=norm_params, norm_method=norm_method,
        )
        all_windows.append(windows)
        all_labels.append(np.full(len(windows), class_idx, dtype=np.int64))
    if len(all_windows) == 0:
        raise ValueError("No test data found")
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y
