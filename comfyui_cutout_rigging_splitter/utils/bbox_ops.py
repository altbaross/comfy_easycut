from __future__ import annotations

import torch


BBox = tuple[int, int, int, int]


def compute_mask_bbox(mask_hw: torch.Tensor, threshold: float = 0.5, padding: int = 0) -> BBox | None:
    if mask_hw.ndim != 2:
        raise ValueError("compute_mask_bbox expects a single [H, W] mask tensor.")

    active = mask_hw > float(threshold)
    nonzero = active.nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return None

    height, width = mask_hw.shape
    y0 = max(int(nonzero[:, 0].min().item()) - padding, 0)
    x0 = max(int(nonzero[:, 1].min().item()) - padding, 0)
    y1 = min(int(nonzero[:, 0].max().item()) + padding + 1, height)
    x1 = min(int(nonzero[:, 1].max().item()) + padding + 1, width)
    return y0, y1, x0, x1


def crop_image_bhwc(image: torch.Tensor, bbox: BBox) -> torch.Tensor:
    y0, y1, x0, x1 = bbox
    return image[:, y0:y1, x0:x1, :]


def crop_mask_bhw(mask: torch.Tensor, bbox: BBox) -> torch.Tensor:
    y0, y1, x0, x1 = bbox
    return mask[:, y0:y1, x0:x1]
