from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BasePoseRefinementBackend(ABC):
    @abstractmethod
    def load(self, device: torch.device) -> None:
        raise NotImplementedError

    @abstractmethod
    def refine(
        self,
        image_bhwc: torch.Tensor,
        part_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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
