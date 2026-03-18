from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.clamp(0.0, 1.0)
    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=radius)
    return dilated.squeeze(1).clamp(0.0, 1.0)


def erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.clamp(0.0, 1.0)
    return 1.0 - dilate_mask(1.0 - mask.clamp(0.0, 1.0), radius)


def feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.clamp(0.0, 1.0)
    kernel_size = radius * 2 + 1
    blurred = F.avg_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=radius)
    return blurred.squeeze(1).clamp(0.0, 1.0)


def refine_logical_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    binary = (mask > 0.5).to(torch.float32)
    if radius <= 0:
        return binary
    closed = erode_mask(dilate_mask(binary, radius), radius)
    return torch.maximum(binary, closed).clamp(0.0, 1.0)


def make_limbs_union_mask(part_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.maximum(
        torch.maximum(part_masks["arm_left"], part_masks["arm_right"]),
        torch.maximum(part_masks["leg_left"], part_masks["leg_right"]),
    )


def make_torso_hole_mask(
    torso_mask: torch.Tensor,
    limbs_union_mask: torch.Tensor,
    padding: int,
    refinement_radius: int = 0,
) -> torch.Tensor:
    if float(torso_mask.max()) == 0.0 or float(limbs_union_mask.max()) == 0.0:
        return torch.zeros_like(torso_mask)
    torso_context = dilate_mask(torso_mask, max(padding, 1))
    limb_context = dilate_mask(limbs_union_mask, min(max(refinement_radius, 0), 1))
    conservative_overlap = (torso_context * limb_context).clamp(0.0, 1.0)
    if refinement_radius > 0:
        conservative_overlap = refine_logical_mask(conservative_overlap, min(refinement_radius, 1))
    return conservative_overlap.clamp(0.0, 1.0)


def _largest_connected_component_numpy(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("largest connected component expects a [H, W] mask.")
    if not mask.any():
        return mask

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_component: list[tuple[int, int]] = []
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []

        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for delta_y, delta_x in neighbors:
                next_y = y + delta_y
                next_x = x + delta_x
                if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                    continue
                if visited[next_y, next_x] or not mask[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                queue.append((next_y, next_x))

        if len(component) > len(best_component):
            best_component = component

    largest = np.zeros_like(mask, dtype=bool)
    for y, x in best_component:
        largest[y, x] = True
    return largest


def keep_largest_connected_component(mask: torch.Tensor) -> torch.Tensor:
    largest_samples: list[torch.Tensor] = []
    for sample_mask in mask:
        binary_np = sample_mask.detach().cpu().numpy() > 0.5
        if not binary_np.any():
            largest_samples.append(torch.zeros_like(sample_mask))
            continue
        largest_np = _largest_connected_component_numpy(binary_np)
        largest_samples.append(torch.as_tensor(largest_np, dtype=torch.float32, device=sample_mask.device))
    return torch.stack(largest_samples, dim=0)


def select_primary_person_masks(part_masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    canonical_union = None
    for mask in part_masks.values():
        canonical_union = mask if canonical_union is None else torch.maximum(canonical_union, mask)
    assert canonical_union is not None

    primary_seed = keep_largest_connected_component(canonical_union)
    primary_mask = dilate_mask(primary_seed, 1)
    return {
        part_name: (part_mask * primary_mask).clamp(0.0, 1.0)
        for part_name, part_mask in part_masks.items()
    }
