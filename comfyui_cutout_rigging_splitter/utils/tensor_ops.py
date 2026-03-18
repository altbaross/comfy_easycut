from __future__ import annotations

import torch
import torch.nn.functional as F


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


def make_part_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return image * mask.unsqueeze(-1)


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a `[B, H, W]` float mask by `radius` pixels using max pooling."""
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=radius)
    return dilated.squeeze(1)


def feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Feather a `[B, H, W]` float mask by averaging within a `radius` pixel window."""
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    blurred = F.avg_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=radius)
    return blurred.squeeze(1).clamp(0.0, 1.0)
