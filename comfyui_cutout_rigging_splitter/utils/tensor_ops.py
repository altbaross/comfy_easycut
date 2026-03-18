from __future__ import annotations

import torch


def ensure_image_bhwc(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor in ComfyUI IMAGE format [B, H, W, 3].")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("image must have shape [B, H, W, 3].")
    if image.dtype != torch.float32:
        image = image.to(torch.float32)
    return image.clamp(0.0, 1.0)


def zeros_mask_like(image: torch.Tensor) -> torch.Tensor:
    return torch.zeros(image.shape[0], image.shape[1], image.shape[2], dtype=torch.float32, device=image.device)


def zeros_image_like(image: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(image, dtype=torch.float32)
