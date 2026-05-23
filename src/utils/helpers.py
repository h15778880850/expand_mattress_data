import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


CLASS_NAMES = ["supine", "lie_on_left", "lie_on_right", "roll_to_left", "roll_to_right"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class Config:
    def __init__(self, path: str):
        with open(path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def __getattr__(self, name):
        if name in self.cfg:
            return self.cfg[name]
        return self.cfg.get(name, {})

    def get(self, *keys, default=None):
        val = self.cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val if val is not None else default

    def to_dict(self):
        return self.cfg


class Logger:
    def __init__(self, log_dir: str, name: str = "train"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{name}_{timestamp}.log"
        self._terminal = sys.stdout

    def write(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        self._terminal.write(log_line + "\n")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    def flush(self):
        self._terminal.flush()
