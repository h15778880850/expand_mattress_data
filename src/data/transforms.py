import numpy as np
import torch


class AddGaussianNoise:
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, scale: float = 0.1, prob: float = 0.5):
        self.scale = scale
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.prob:
            return x
        factor = 1.0 + np.random.uniform(-self.scale, self.scale)
        return x * factor
