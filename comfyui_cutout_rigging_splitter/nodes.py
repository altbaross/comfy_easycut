from __future__ import annotations

import torch

from .backends import BaseHumanParsingBackend, TransformersHumanParsingBackend
from .constants import CANONICAL_PARTS, RETURN_NAMES, RETURN_TYPES
from .utils import dilate_mask, ensure_image_bhwc, feather_mask, make_part_image, zeros_mask_like


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

    def _part_masks_from_labels(self, label_masks: list[torch.Tensor], image: torch.Tensor) -> dict[str, torch.Tensor]:
        if not hasattr(self.backend, "label_id_to_part"):
            raise RuntimeError("Human parsing backend is missing the required 'label_id_to_part' mapping.")
        part_masks = {
            part: zeros_mask_like(image)
            for part in CANONICAL_PARTS
        }
        label_id_to_part = {
            int(label_id): part_name
            for label_id, part_name in getattr(self.backend, "label_id_to_part", {}).items()
            if part_name in CANONICAL_PARTS
        }

        for batch_index, label_mask in enumerate(label_masks):
            sample_mask = label_mask.to(device=image.device, dtype=torch.int64)
            sample_parts = {
                part: torch.zeros_like(sample_mask, dtype=torch.float32)
                for part in CANONICAL_PARTS
            }
            for label_id in torch.unique(sample_mask):
                part_name = label_id_to_part.get(int(label_id.item()))
                if part_name is None:
                    continue
                sample_parts[part_name] = torch.maximum(
                    sample_parts[part_name],
                    (sample_mask == label_id).to(torch.float32),
                )

            for part_name, sample_part_mask in sample_parts.items():
                part_masks[part_name][batch_index] = sample_part_mask

        return part_masks

    def _coerce_label_masks(
        self,
        label_arrays: list[object],
        image: torch.Tensor,
    ) -> list[torch.Tensor]:
        expected_shape = (image.shape[1], image.shape[2])
        label_masks: list[torch.Tensor] = []
        for batch_index, label_array in enumerate(label_arrays):
            label_mask = torch.as_tensor(label_array, dtype=torch.int64, device=image.device)
            if label_mask.ndim != 2 or tuple(label_mask.shape) != expected_shape:
                raise RuntimeError(
                    "Human parsing backend returned an invalid label mask shape for batch index "
                    f"{batch_index}: expected {expected_shape}, received {tuple(label_mask.shape)}."
                )
            label_masks.append(label_mask)
        return label_masks

    def process(
        self,
        image: torch.Tensor,
        feathering_amount: int = 2,
        padding: int = 8,
    ) -> tuple[torch.Tensor, ...]:
        """Split a ComfyUI IMAGE tensor into canonical cutout-rigging part images and masks."""
        image = ensure_image_bhwc(image)
        if not isinstance(feathering_amount, int) or not 0 <= feathering_amount <= 16:
            raise ValueError("feathering_amount must be an integer in the range [0, 16].")
        if not isinstance(padding, int) or not 0 <= padding <= 128:
            raise ValueError("padding must be an integer in the range [0, 128].")

        label_arrays = self.backend.infer(image)
        if not isinstance(label_arrays, list):
            raise RuntimeError("Human parsing backend must return a list of per-sample label masks.")
        if len(label_arrays) != image.shape[0]:
            raise RuntimeError(
                "Human parsing backend returned an unexpected number of label masks: "
                f"expected {image.shape[0]}, received {len(label_arrays)}."
            )

        label_masks = self._coerce_label_masks(label_arrays, image)
        logical_part_masks = self._part_masks_from_labels(label_masks, image)
        output_part_masks = {
            part_name: feather_mask(part_mask, feathering_amount)
            for part_name, part_mask in logical_part_masks.items()
        }

        part_images = {
            part_name: make_part_image(image, part_mask)
            for part_name, part_mask in output_part_masks.items()
        }

        limbs_union_mask = torch.maximum(
            torch.maximum(logical_part_masks["arm_left"], logical_part_masks["arm_right"]),
            torch.maximum(logical_part_masks["leg_left"], logical_part_masks["leg_right"]),
        )
        torso_hole_mask = limbs_union_mask * dilate_mask(logical_part_masks["torso"], padding)

        return (
            part_images["head"],
            output_part_masks["head"],
            part_images["torso"],
            output_part_masks["torso"],
            part_images["arm_left"],
            output_part_masks["arm_left"],
            part_images["arm_right"],
            output_part_masks["arm_right"],
            part_images["leg_left"],
            output_part_masks["leg_left"],
            part_images["leg_right"],
            output_part_masks["leg_right"],
            limbs_union_mask,
            torso_hole_mask,
        )
