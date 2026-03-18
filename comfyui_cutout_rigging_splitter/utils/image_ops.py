from __future__ import annotations

import torch

from ..constants import DEFAULT_EMPTY_CROP_SIZE
from .bbox_ops import BBox, crop_image_bhwc, crop_mask_bhw
from .fallback import empty_crop_image, empty_crop_mask


def make_part_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return image * mask.unsqueeze(-1)


def crop_part_image_and_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: BBox | None,
    *,
    empty_size: int = DEFAULT_EMPTY_CROP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bbox is None:
        return empty_crop_image(image.device, image.dtype, size=empty_size), empty_crop_mask(mask.device, size=empty_size)
    return crop_image_bhwc(image, bbox), crop_mask_bhw(mask, bbox)
