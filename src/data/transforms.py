import numpy as np
import torch
from typing import Optional


class AddGaussianNoise:
    def __init__(self, std: float = 0.01, prob: float = 0.5):
        self.std = std
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.prob:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise


class TimeWarp:
    def __init__(self, prob: float = 0.5, max_warp: float = 0.2):
        self.prob = prob
        self.max_warp = max_warp

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.prob:
            return x
        C, L = x.shape
        orig = np.linspace(0, L - 1, L)
        shift = np.random.uniform(-self.max_warp, self.max_warp, L).cumsum()
        new = orig + shift
        new = np.clip(new, 0, L - 1)
        warped = np.zeros_like(x.numpy())
        for c in range(C):
            warped[c] = np.interp(orig, new, x.numpy()[c])
        return torch.from_numpy(warped).float()


class AmplitudeShift:
    def __init__(self, scale_min: float = 0.95, scale_max: float = 1.05, prob: float = 0.3):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.prob:
            return x
        factor = np.random.uniform(self.scale_min, self.scale_max)
        return x * factor


class Compose:
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def build_transform(config) -> Optional[Compose]:
    aug = config.get("augmentation", default={})
    if not aug:
        return None

    t_list = []

    gn = aug.get("gaussian_noise", {})
    if gn.get("prob", 0) > 0:
        t_list.append(AddGaussianNoise(std=gn.get("std", 0.01), prob=gn.get("prob", 0.5)))

    amp = aug.get("amplitude_scale", {})
    if amp.get("prob", 0) > 0:
        t_list.append(AmplitudeShift(
            scale_min=amp.get("scale_min", 0.95),
            scale_max=amp.get("scale_max", 1.05),
            prob=amp.get("prob", 0.3),
        ))

    tw = aug.get("time_warp", {})
    if tw.get("prob", 0) > 0:
        t_list.append(TimeWarp(prob=tw.get("prob", 0.1), max_warp=tw.get("max_warp", 0.1)))

    return Compose(t_list) if t_list else None
