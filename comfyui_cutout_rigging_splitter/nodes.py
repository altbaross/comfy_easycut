from __future__ import annotations

import torch

from .backends import (
    BaseHumanParsingBackend,
    BasePoseRefinementBackend,
    GoogleNanoBananaParsingBackend,
    build_human_parsing_backend_from_environment,
)
from .backends.google_nano_banana_parsing import (
    GOOGLE_NANO_BANANA_API_BASE,
    GOOGLE_NANO_BANANA_MODEL_ID,
    GOOGLE_NANO_BANANA_TIMEOUT_SECONDS,
)
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
    dilate_mask,
    ensure_image_bhwc,
    feather_mask,
    make_limbs_union_mask,
    make_part_image,
    make_torso_hole_mask,
    refine_logical_mask,
    select_primary_person_masks,
    split_mask_left_right,
    zeros_mask_like,
)

_SPECIAL_LABEL_PARTS = frozenset({"pants"})
_UPPER_CLOTHES_LABEL_ID = 4
_EYE_BAND_TOP_RATIO = 0.2
_EYE_BAND_BOTTOM_RATIO = 0.55
_LEFT_EYE_X0_RATIO = 0.1
_LEFT_EYE_X1_RATIO = 0.4
_RIGHT_EYE_X0_RATIO = 0.6
_RIGHT_EYE_X1_RATIO = 0.9
_BACKEND_CONNECTOR_TYPE = "EASYCUT_HUMAN_PARSING_BACKEND"


def _normalize_label_name(label_name: object) -> str:
    return str(label_name).strip().lower().replace("_", "-").replace(" ", "-")


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
        self.backend = backend or build_human_parsing_backend_from_environment()
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
                "human_parsing_backend": (_BACKEND_CONNECTOR_TYPE,),
            },
        }

    def _resolve_backend(self, backend: BaseHumanParsingBackend | None = None) -> BaseHumanParsingBackend:
        active_backend = backend or self.backend
        if not isinstance(active_backend, BaseHumanParsingBackend):
            raise RuntimeError("human_parsing_backend input must provide a BaseHumanParsingBackend instance.")
        return active_backend

    def _part_masks_from_labels(
        self,
        label_masks: list[torch.Tensor],
        image: torch.Tensor,
        backend: BaseHumanParsingBackend | None = None,
    ) -> dict[str, torch.Tensor]:
        active_backend = self._resolve_backend(backend)
        if not hasattr(active_backend, "label_id_to_part"):
            raise RuntimeError("Human parsing backend is missing the required 'label_id_to_part' mapping.")
        if not isinstance(getattr(active_backend, "label_id_to_part"), dict):
            raise RuntimeError("Human parsing backend 'label_id_to_part' mapping must be a dictionary.")

        part_masks = {part: zeros_mask_like(image) for part in CANONICAL_PARTS}
        label_id_to_part = {
            int(label_id): str(part_name)
            for label_id, part_name in getattr(active_backend, "label_id_to_part", {}).items()
            if str(part_name) in CANONICAL_PARTS or str(part_name) in _SPECIAL_LABEL_PARTS
        }

        for batch_index, label_mask in enumerate(label_masks):
            sample_mask = label_mask.to(device=image.device, dtype=torch.int64)
            for label_id, part_name in label_id_to_part.items():
                sample_part_mask = (sample_mask == label_id).to(torch.float32)
                if float(sample_part_mask.max()) == 0.0:
                    continue
                if part_name == "pants":
                    leg_left_mask, leg_right_mask = split_mask_left_right(sample_part_mask)
                    part_masks["leg_left"][batch_index] = torch.maximum(part_masks["leg_left"][batch_index], leg_left_mask)
                    part_masks["leg_right"][batch_index] = torch.maximum(part_masks["leg_right"][batch_index], leg_right_mask)
                    continue
                if part_name not in CANONICAL_PARTS:
                    continue
                part_masks[part_name][batch_index] = torch.maximum(part_masks[part_name][batch_index], sample_part_mask)

        return part_masks

    def _label_ids_for_names(
        self,
        *label_names: str,
        backend: BaseHumanParsingBackend | None = None,
    ) -> set[int]:
        active_backend = self._resolve_backend(backend)
        normalized_targets = {_normalize_label_name(label_name) for label_name in label_names}
        return {
            int(label_id)
            for label_id, label_name in getattr(active_backend, "id_to_label", {}).items()
            if _normalize_label_name(label_name) in normalized_targets
        }

    @staticmethod
    def _make_eye_mask(face_mask: torch.Tensor) -> torch.Tensor:
        if face_mask.ndim != 2:
            raise ValueError("_make_eye_mask expects a [H, W] face mask.")

        binary = face_mask > 0.5
        if not bool(binary.any()):
            return torch.zeros_like(face_mask, dtype=torch.float32)

        nonzero = torch.nonzero(binary, as_tuple=False)
        y0 = int(nonzero[:, 0].min().item())
        y1 = int(nonzero[:, 0].max().item()) + 1
        x0 = int(nonzero[:, 1].min().item())
        x1 = int(nonzero[:, 1].max().item()) + 1
        height = y1 - y0
        width = x1 - x0
        # Require at least 2 rows for a visible eye band and 3 columns so the
        # left/right eye windows can remain distinct instead of collapsing into
        # a single block on very small face masks.
        if height < 2 or width < 3:
            return torch.zeros_like(face_mask, dtype=torch.float32)

        # For 2D illustration faces, eyes typically sit in the upper-middle face
        # band with a small center gap for the nose bridge. These proportional
        # windows intentionally bias toward that anime/cutout layout.
        eye_band_y0 = y0 + int(round(height * _EYE_BAND_TOP_RATIO))
        min_eye_band_height = eye_band_y0 - y0 + 1
        # Always keep at least one scanline in the eye band even for short face
        # masks so that small illustration faces can still produce an eye layer.
        eye_band_y1 = min(y1, y0 + max(int(round(height * _EYE_BAND_BOTTOM_RATIO)), min_eye_band_height))
        left_eye_x0 = x0 + int(round(width * _LEFT_EYE_X0_RATIO))
        left_eye_x1 = min(x1, x0 + max(int(round(width * _LEFT_EYE_X1_RATIO)), left_eye_x0 - x0 + 1))
        # The right-eye window is clamped to start after the computed left-eye
        # window so the two eye regions stay separated even on narrow face masks.
        right_eye_x0 = min(x1 - 1, x0 + max(int(round(width * _RIGHT_EYE_X0_RATIO)), left_eye_x1 - x0 + 1))
        right_eye_x1 = min(x1, x0 + max(int(round(width * _RIGHT_EYE_X1_RATIO)), right_eye_x0 - x0 + 1))

        eye_windows = torch.zeros_like(face_mask, dtype=torch.float32)
        eye_windows[eye_band_y0:eye_band_y1, left_eye_x0:left_eye_x1] = 1.0
        eye_windows[eye_band_y0:eye_band_y1, right_eye_x0:right_eye_x1] = 1.0
        return (eye_windows * binary.to(torch.float32)).clamp(0.0, 1.0)

    def _derive_illustration_part_masks(
        self,
        label_masks: list[torch.Tensor],
        part_masks: dict[str, torch.Tensor],
        backend: BaseHumanParsingBackend | None = None,
    ) -> dict[str, torch.Tensor]:
        face_label_ids = self._label_ids_for_names("face", backend=backend)
        if not face_label_ids:
            return part_masks

        for batch_index, label_mask in enumerate(label_masks):
            sample_mask = label_mask.to(device=part_masks["head"].device, dtype=torch.int64)
            face_mask = torch.zeros_like(part_masks["head"][batch_index])
            for label_id in face_label_ids:
                face_mask = torch.maximum(face_mask, (sample_mask == label_id).to(torch.float32))
            if float(face_mask.max()) == 0.0:
                continue

            eyes_mask = self._make_eye_mask(face_mask)
            part_masks["eyes"][batch_index] = eyes_mask
            if float(part_masks["hair"][batch_index].max()) > 0.0:
                part_masks["head"][batch_index] = (
                    part_masks["head"][batch_index] * (1.0 - part_masks["hair"][batch_index])
                ).clamp(0.0, 1.0)
            if float(eyes_mask.max()) > 0.0:
                part_masks["head"][batch_index] = (part_masks["head"][batch_index] * (1.0 - eyes_mask)).clamp(0.0, 1.0)

        return part_masks

    def _redistribute_garments_to_limbs(
        self,
        label_masks: list[torch.Tensor],
        part_masks: dict[str, torch.Tensor],
        garment_label_id: int = _UPPER_CLOTHES_LABEL_ID,
        expansion_radius: int = 1,
    ) -> dict[str, torch.Tensor]:
        torso_masks = part_masks["torso"]
        for batch_index, label_mask in enumerate(label_masks):
            garment_mask = (label_mask.to(device=torso_masks.device, dtype=torch.int64) == garment_label_id).to(torch.float32).unsqueeze(0)
            if float(garment_mask.max()) == 0.0:
                continue
            for arm_key in ("arm_left", "arm_right"):
                arm_mask = part_masks[arm_key][batch_index].unsqueeze(0)
                if float(arm_mask.max()) == 0.0:
                    continue
                sleeve_region = (dilate_mask(arm_mask, expansion_radius) * garment_mask).clamp(0.0, 1.0).squeeze(0)
                if float(sleeve_region.max()) == 0.0:
                    continue
                part_masks[arm_key][batch_index] = torch.maximum(part_masks[arm_key][batch_index], sleeve_region)
                part_masks["torso"][batch_index] = (part_masks["torso"][batch_index] * (1.0 - sleeve_region)).clamp(0.0, 1.0)
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

    def _load_and_infer(
        self,
        image: torch.Tensor,
        backend: BaseHumanParsingBackend | None = None,
    ) -> list[object]:
        active_backend = self._resolve_backend(backend)
        try:
            backend_load = active_backend.load
        except AttributeError as exc:
            raise RuntimeError("Human parsing backend is missing the required load(device: torch.device) method.") from exc
        if not callable(backend_load):
            raise RuntimeError(
                "Human parsing backend load attribute must be callable and accept a torch.device parameter."
            )
        backend_load(image.device)

        try:
            backend_infer = active_backend.infer
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
            part_bbox = compute_mask_bbox(
                logical_part_masks[part_name][0],
                threshold=mask_threshold,
                padding=crop_padding,
            )
            self.last_crop_boxes[part_name] = part_bbox
            cropped_images[part_name], cropped_masks[part_name] = crop_part_image_and_mask(
                part_images[part_name],
                part_masks[part_name],
                part_bbox,
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
        human_parsing_backend: BaseHumanParsingBackend | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Split a ComfyUI IMAGE tensor into canonical cutout-rigging part images and masks."""
        image = ensure_image_bhwc(image)
        self._validate_parameters(feathering_amount, padding, crop_padding, morphology_strength, mask_threshold)

        label_arrays = self._load_and_infer(image, backend=human_parsing_backend)
        label_masks = self._coerce_label_masks(label_arrays, image)
        logical_part_masks = self._part_masks_from_labels(label_masks, image, backend=human_parsing_backend)
        logical_part_masks = self._derive_illustration_part_masks(
            label_masks,
            logical_part_masks,
            backend=human_parsing_backend,
        )
        logical_part_masks = self._redistribute_garments_to_limbs(label_masks, logical_part_masks)
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
            part_images["eyes"],
            output_part_masks["eyes"],
            part_images["hair"],
            output_part_masks["hair"],
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


class GoogleNanoBananaConnector:
    CATEGORY = "CutoutAnimation/Connectors"
    FUNCTION = "build_backend"
    RETURN_TYPES = (_BACKEND_CONNECTOR_TYPE,)
    RETURN_NAMES = ("human_parsing_backend",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "model_id": ("STRING", {"default": GOOGLE_NANO_BANANA_MODEL_ID, "multiline": False}),
                "api_base": ("STRING", {"default": GOOGLE_NANO_BANANA_API_BASE, "multiline": False}),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": GOOGLE_NANO_BANANA_TIMEOUT_SECONDS, "min": 1.0, "max": 600.0, "step": 1.0},
                ),
            },
        }

    def build_backend(
        self,
        api_key: str,
        model_id: str = GOOGLE_NANO_BANANA_MODEL_ID,
        api_base: str = GOOGLE_NANO_BANANA_API_BASE,
        timeout_seconds: float = GOOGLE_NANO_BANANA_TIMEOUT_SECONDS,
    ) -> tuple[BaseHumanParsingBackend]:
        backend = GoogleNanoBananaParsingBackend(
            api_key=str(api_key).strip(),
            model_id=str(model_id).strip() or GOOGLE_NANO_BANANA_MODEL_ID,
            api_base=str(api_base).strip() or GOOGLE_NANO_BANANA_API_BASE,
            timeout_seconds=float(timeout_seconds),
        )
        return (backend,)
