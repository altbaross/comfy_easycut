from __future__ import annotations

import torch


def empty_crop_image(device: torch.device, dtype: torch.dtype, size: int = 1) -> torch.Tensor:
    return torch.zeros((1, size, size, 3), dtype=dtype, device=device)


def empty_crop_mask(device: torch.device, size: int = 1) -> torch.Tensor:
    return torch.zeros((1, size, size), dtype=torch.float32, device=device)
