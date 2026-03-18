from __future__ import annotations

import torch

from .backends import BaseHumanParsingBackend, BasePoseRefinementBackend, TransformersHumanParsingBackend
from .constants import (
    CANONICAL_PARTS,
    DEFAULT_MASK_THRESHOLD,
    MAX_FEATHERING_AMOUNT,
    MAX_MORPHOLOGY_STRENGTH,
    MAX_PADDING,
    RETURN_NAMES,
    RETURN_TYPES,
)
from .utils import (
    compute_mask_bbox,
    crop_part_image_and_mask,
    ensure_image_bhwc,
    feather_mask,
    make_limbs_union_mask,
    make_part_image,
    make_torso_hole_mask,
    refine_logical_mask,
    select_primary_person_masks,
    zeros_mask_like,
)


class CutoutRiggingSplitter:
    CATEGORY = "CutoutAnimation/Processing"
    FUNCTION = "process"
    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES

    def __init__(
        self,
        backend: BaseHumanParsingBackend | None = None,
        pose_refiner: BasePoseRefinementBackend | None = None,
    ) -> None:
        self.backend = backend or TransformersHumanParsingBackend()
        self.pose_refiner = pose_refiner
        self.last_crop_boxes: dict[str, tuple[int, int, int, int] | None] = {}

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "feathering_amount": ("INT", {"default": 2, "min": 0, "max": MAX_FEATHERING_AMOUNT, "step": 1}),
                "padding": ("INT", {"default": 8, "min": 0, "max": MAX_PADDING, "step": 1}),
            },
            "optional": {
                "crop_mode": ("BOOLEAN", {"default": False}),
                "crop_padding": ("INT", {"default": 8, "min": 0, "max": MAX_PADDING, "step": 1}),
                "morphology_strength": ("INT", {"default": 0, "min": 0, "max": MAX_MORPHOLOGY_STRENGTH, "step": 1}),
                "mask_threshold": ("FLOAT", {"default": DEFAULT_MASK_THRESHOLD, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_pose_refinement": ("BOOLEAN", {"default": False}),
            },
        }

    def _part_masks_from_labels(self, label_masks: list[torch.Tensor], image: torch.Tensor) -> dict[str, torch.Tensor]:
        if not hasattr(self.backend, "label_id_to_part"):
            raise RuntimeError("Human parsing backend is missing the required 'label_id_to_part' mapping.")
        if not isinstance(getattr(self.backend, "label_id_to_part"), dict):
            raise RuntimeError("Human parsing backend 'label_id_to_part' mapping must be a dictionary.")

        part_masks = {part: zeros_mask_like(image) for part in CANONICAL_PARTS}
        label_id_to_part = {
            int(label_id): part_name
            for label_id, part_name in getattr(self.backend, "label_id_to_part", {}).items()
            if part_name in CANONICAL_PARTS
        }

        for batch_index, label_mask in enumerate(label_masks):
            sample_mask = label_mask.to(device=image.device, dtype=torch.int64)
            for label_id, part_name in label_id_to_part.items():
                sample_part_mask = (sample_mask == label_id).to(torch.float32)
                if float(sample_part_mask.max()) == 0.0:
                    continue
                part_masks[part_name][batch_index] = torch.maximum(part_masks[part_name][batch_index], sample_part_mask)

        return part_masks

    def _coerce_label_masks(self, label_arrays: list[object], image: torch.Tensor) -> list[torch.Tensor]:
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

    def _maybe_refine_with_pose(
        self,
        image: torch.Tensor,
        part_masks: dict[str, torch.Tensor],
        enable_pose_refinement: bool,
    ) -> dict[str, torch.Tensor]:
        if not enable_pose_refinement or self.pose_refiner is None:
            return part_masks
        self.pose_refiner.load(image.device)
        refined_masks = self.pose_refiner.refine(image, dict(part_masks))
        if not isinstance(refined_masks, dict):
            raise RuntimeError("Pose refinement backend must return a dictionary of canonical part masks.")
        return {
            part_name: torch.as_tensor(refined_masks.get(part_name, part_masks[part_name]), device=image.device, dtype=torch.float32)
            for part_name in CANONICAL_PARTS
        }

    def _validate_parameters(
        self,
        feathering_amount: int,
        padding: int,
        crop_padding: int,
        morphology_strength: int,
        mask_threshold: float,
    ) -> None:
        if not isinstance(feathering_amount, int) or not 0 <= feathering_amount <= MAX_FEATHERING_AMOUNT:
            raise ValueError(f"feathering_amount must be an integer in the range [0, {MAX_FEATHERING_AMOUNT}].")
        if not isinstance(padding, int) or not 0 <= padding <= MAX_PADDING:
            raise ValueError(f"padding must be an integer in the range [0, {MAX_PADDING}].")
        if not isinstance(crop_padding, int) or not 0 <= crop_padding <= MAX_PADDING:
            raise ValueError(f"crop_padding must be an integer in the range [0, {MAX_PADDING}].")
        if not isinstance(morphology_strength, int) or not 0 <= morphology_strength <= MAX_MORPHOLOGY_STRENGTH:
            raise ValueError(
                f"morphology_strength must be an integer in the range [0, {MAX_MORPHOLOGY_STRENGTH}]."
            )
        if not isinstance(mask_threshold, (float, int)) or not 0.0 <= float(mask_threshold) <= 1.0:
            raise ValueError("mask_threshold must be a float in the range [0.0, 1.0].")

    def _load_and_infer(self, image: torch.Tensor) -> list[object]:
        try:
            backend_load = self.backend.load
        except AttributeError as exc:
            raise RuntimeError("Human parsing backend is missing the required load(device: torch.device) method.") from exc
        if not callable(backend_load):
            raise RuntimeError(
                "Human parsing backend load attribute must be callable and accept a torch.device parameter."
            )
        backend_load(image.device)

        try:
            backend_infer = self.backend.infer
        except AttributeError as exc:
            raise RuntimeError(
                "Human parsing backend is missing the required infer(image_bhwc: torch.Tensor) method."
            ) from exc
        if not callable(backend_infer):
            raise RuntimeError(
                "Human parsing backend infer attribute must be callable and accept an image_bhwc: torch.Tensor parameter."
            )
        label_arrays = backend_infer(image)
        if not isinstance(label_arrays, list):
            raise RuntimeError("Human parsing backend must return a list of per-sample label masks.")
        if len(label_arrays) != image.shape[0]:
            raise RuntimeError(
                "Human parsing backend returned an unexpected number of label masks: "
                f"expected {image.shape[0]}, received {len(label_arrays)}."
            )
        return label_arrays

    def _crop_outputs(
        self,
        part_images: dict[str, torch.Tensor],
        logical_part_masks: dict[str, torch.Tensor],
        part_masks: dict[str, torch.Tensor],
        limbs_union_mask: torch.Tensor,
        torso_hole_mask: torch.Tensor,
        crop_mode: bool,
        crop_padding: int,
        mask_threshold: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        self.last_crop_boxes = {}
        if not crop_mode or next(iter(part_images.values())).shape[0] != 1:
            return part_images, part_masks, limbs_union_mask, torso_hole_mask

        cropped_images: dict[str, torch.Tensor] = {}
        cropped_masks: dict[str, torch.Tensor] = {}
        for part_name in CANONICAL_PARTS:
            bbox = compute_mask_bbox(logical_part_masks[part_name][0], threshold=mask_threshold, padding=crop_padding)
            self.last_crop_boxes[part_name] = bbox
            cropped_images[part_name], cropped_masks[part_name] = crop_part_image_and_mask(
                part_images[part_name],
                part_masks[part_name],
                bbox,
            )

        limbs_bbox = compute_mask_bbox(limbs_union_mask[0], threshold=mask_threshold, padding=crop_padding)
        torso_hole_bbox = compute_mask_bbox(torso_hole_mask[0], threshold=mask_threshold, padding=crop_padding)
        self.last_crop_boxes["limbs_union"] = limbs_bbox
        self.last_crop_boxes["torso_hole"] = torso_hole_bbox
        _, cropped_limbs_union = crop_part_image_and_mask(
            part_images["torso"],
            limbs_union_mask,
            limbs_bbox,
        )
        _, cropped_torso_hole = crop_part_image_and_mask(
            part_images["torso"],
            torso_hole_mask,
            torso_hole_bbox,
        )
        return cropped_images, cropped_masks, cropped_limbs_union, cropped_torso_hole

    def process(
        self,
        image: torch.Tensor,
        feathering_amount: int = 2,
        padding: int = 8,
        crop_mode: bool = False,
        crop_padding: int = 8,
        morphology_strength: int = 0,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        enable_pose_refinement: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Split a ComfyUI IMAGE tensor into canonical cutout-rigging part images and masks."""
        image = ensure_image_bhwc(image)
        self._validate_parameters(feathering_amount, padding, crop_padding, morphology_strength, mask_threshold)

        label_arrays = self._load_and_infer(image)
        label_masks = self._coerce_label_masks(label_arrays, image)
        logical_part_masks = self._part_masks_from_labels(label_masks, image)
        logical_part_masks = select_primary_person_masks(logical_part_masks)
        logical_part_masks = self._maybe_refine_with_pose(image, logical_part_masks, enable_pose_refinement)
        logical_part_masks = {
            part_name: refine_logical_mask(part_mask, morphology_strength)
            for part_name, part_mask in logical_part_masks.items()
        }

        output_part_masks = {
            part_name: feather_mask(part_mask, feathering_amount)
            for part_name, part_mask in logical_part_masks.items()
        }
        part_images = {
            part_name: make_part_image(image, part_mask)
            for part_name, part_mask in output_part_masks.items()
        }

        limbs_union_mask = make_limbs_union_mask(logical_part_masks)
        torso_hole_mask = make_torso_hole_mask(
            logical_part_masks["torso"],
            limbs_union_mask,
            padding,
            morphology_strength,
        )

        part_images, output_part_masks, limbs_union_mask, torso_hole_mask = self._crop_outputs(
            part_images,
            logical_part_masks,
            output_part_masks,
            limbs_union_mask,
            torso_hole_mask,
            crop_mode,
            crop_padding,
            float(mask_threshold),
        )

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
