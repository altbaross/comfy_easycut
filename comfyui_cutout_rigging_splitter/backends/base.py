from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseHumanParsingBackend(ABC):
    id_to_label: dict[int, str]

    def __init__(self) -> None:
        self.id_to_label = {}

    @abstractmethod
    def load(self, device: torch.device) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, image_bhwc: torch.Tensor) -> list[np.ndarray]:
        """
        Args:
            image_bhwc: [B, H, W, 3], float32 in [0, 1]
        Returns:
            List of label index masks with shape [H, W] and dtype int32.
        """
        raise NotImplementedError
