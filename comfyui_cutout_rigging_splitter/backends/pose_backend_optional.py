from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BasePoseRefinementBackend(ABC):
    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load any pose-refinement resources onto ``device`` before ``refine`` is called."""
        raise NotImplementedError

    @abstractmethod
    def refine(
        self,
        image_bhwc: torch.Tensor,
        part_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Refine canonical part masks for a ComfyUI ``IMAGE`` batch.

        Args:
            image_bhwc: Input image tensor with shape ``[B, H, W, 3]``.
            part_masks: Canonical part masks keyed by ``head``, ``eyes``,
                ``hair``, ``torso``, ``arm_left``, ``arm_right``, ``leg_left``,
                and ``leg_right``. Each tensor must have shape ``[B, H, W]``
                and values in ``[0, 1]``.

        Returns:
            A dictionary with the same canonical keys and mask tensor shapes.
            Implementations should preserve batch size and spatial dimensions.
        """
        raise NotImplementedError


class NoOpPoseRefinementBackend(BasePoseRefinementBackend):
    def load(self, device: torch.device) -> None:
        del device

    def refine(
        self,
        image_bhwc: torch.Tensor,
        part_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del image_bhwc
        return dict(part_masks)
