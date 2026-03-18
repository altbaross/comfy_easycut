from __future__ import annotations

from typing import Iterable

import torch

from .backends import BaseHumanParsingBackend, TransformersHumanParsingBackend
from .constants import CANONICAL_PARTS, RETURN_NAMES, RETURN_TYPES
from .utils import ensure_image_bhwc, make_part_image, zeros_mask_like


def _normalize_label_name(value: object) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


LIP_STYLE_LABEL_TO_PART = {
    "hat": "head",
    "hair": "head",
    "face": "head",
    "head": "head",
    "sunglasses": "head",
    "upper-clothes": "torso",
    "upperclothes": "torso",
    "coat": "torso",
    "dress": "torso",
    "scarf": "torso",
    "jumpsuits": "torso",
    "jumpsuit": "torso",
    "torso": "torso",
    "neck": "torso",
    "left-arm": "arm_left",
    "left hand": "arm_left",
    "left-hand": "arm_left",
    "lefthand": "arm_left",
    "right-arm": "arm_right",
    "right hand": "arm_right",
    "right-hand": "arm_right",
    "righthand": "arm_right",
    "left-leg": "leg_left",
    "left shoe": "leg_left",
    "left-shoe": "leg_left",
    "left-foot": "leg_left",
    "leftfoot": "leg_left",
    "pants-left": "leg_left",
    "right-leg": "leg_right",
    "right shoe": "leg_right",
    "right-shoe": "leg_right",
    "right-foot": "leg_right",
    "rightfoot": "leg_right",
    "pants-right": "leg_right",
}


class CutoutRiggingSplitter:
    CATEGORY = "CutoutAnimation/Processing"
    FUNCTION = "process"
    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES

    def __init__(self, backend: BaseHumanParsingBackend | None = None) -> None:
        self.backend = backend or TransformersHumanParsingBackend()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "feathering_amount": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "padding": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
            }
        }

    def _part_masks_from_labels(self, label_masks: Iterable[torch.Tensor], image: torch.Tensor) -> dict[str, torch.Tensor]:
        part_masks = {
            part: zeros_mask_like(image)
            for part in CANONICAL_PARTS
        }

        id_to_label = {
            int(label_id): _normalize_label_name(label_name)
            for label_id, label_name in getattr(self.backend, "id_to_label", {}).items()
        }

        for batch_index, label_mask in enumerate(label_masks):
            sample_mask = label_mask.to(device=image.device, dtype=torch.int64)
            sample_parts = {
                part: torch.zeros_like(sample_mask, dtype=torch.float32)
                for part in CANONICAL_PARTS
            }
            for label_id in torch.unique(sample_mask):
                label_name = id_to_label.get(int(label_id.item()))
                if label_name is None:
                    continue
                part_name = LIP_STYLE_LABEL_TO_PART.get(label_name)
                if part_name is None:
                    continue
                sample_parts[part_name] = torch.maximum(
                    sample_parts[part_name],
                    (sample_mask == label_id).to(torch.float32),
                )

            for part_name, sample_part_mask in sample_parts.items():
                part_masks[part_name][batch_index] = sample_part_mask

        return part_masks

    def process(self, image: torch.Tensor, feathering_amount: int = 2, padding: int = 8):
        del feathering_amount, padding
        image = ensure_image_bhwc(image)

        label_arrays = self.backend.infer(image)
        if len(label_arrays) != image.shape[0]:
            raise RuntimeError(
                "Human parsing backend returned an unexpected number of label masks: "
                f"expected {image.shape[0]}, received {len(label_arrays)}."
            )

        label_masks = [
            torch.as_tensor(label_array, dtype=torch.int64, device=image.device)
            for label_array in label_arrays
        ]
        part_masks = self._part_masks_from_labels(label_masks, image)

        part_images = {
            part_name: make_part_image(image, part_mask)
            for part_name, part_mask in part_masks.items()
        }

        limbs_union_mask = torch.maximum(
            torch.maximum(part_masks["arm_left"], part_masks["arm_right"]),
            torch.maximum(part_masks["leg_left"], part_masks["leg_right"]),
        )
        torso_hole_mask = limbs_union_mask.clone()

        return (
            part_images["head"],
            part_masks["head"],
            part_images["torso"],
            part_masks["torso"],
            part_images["arm_left"],
            part_masks["arm_left"],
            part_images["arm_right"],
            part_masks["arm_right"],
            part_images["leg_left"],
            part_masks["leg_left"],
            part_images["leg_right"],
            part_masks["leg_right"],
            limbs_union_mask,
            torso_hole_mask,
        )
