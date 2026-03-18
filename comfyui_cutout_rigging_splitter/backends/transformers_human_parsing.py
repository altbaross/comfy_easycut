from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseHumanParsingBackend


DEFAULT_MODEL_ID = os.getenv(
    "COMFY_EASYCUT_HUMAN_PARSING_MODEL",
    "mattmdjaga/segformer_b2_clothes",
)

DEFAULT_MODEL_ID_TO_LABEL = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper-clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left-shoe",
    10: "right-shoe",
    11: "face",
    12: "left-leg",
    13: "right-leg",
    14: "left-arm",
    15: "right-arm",
    16: "bag",
    17: "scarf",
}

DEFAULT_MODEL_LABEL_ID_TO_PART = {
    1: "head",
    2: "head",
    3: "head",
    4: "torso",
    7: "torso",
    9: "leg_left",
    10: "leg_right",
    11: "head",
    12: "leg_left",
    13: "leg_right",
    14: "arm_left",
    15: "arm_right",
    17: "torso",
}

DEFAULT_MODEL_LABEL_TO_PART = {
    DEFAULT_MODEL_ID_TO_LABEL[label_id]: part_name
    for label_id, part_name in DEFAULT_MODEL_LABEL_ID_TO_PART.items()
}


def _normalize_label_name(value: object) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


class TransformersHumanParsingBackend(BaseHumanParsingBackend):
    """Lazy-loaded semantic human parsing backend backed by Hugging Face transformers.

    Args:
        model_id: Model identifier passed to ``from_pretrained`` for both the
            image processor and semantic segmentation model.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        super().__init__()
        self.model_id = model_id
        self._device: torch.device | None = None
        self._image_processor = None
        self._model = None

    def load(self, device: torch.device) -> None:
        if self._model is not None and self._device == device:
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "CutoutRiggingSplitter requires the optional 'transformers' package "
                f"for human parsing backend '{self.model_id}'."
            ) from exc

        try:
            image_processor = AutoImageProcessor.from_pretrained(self.model_id)
            model = AutoModelForSemanticSegmentation.from_pretrained(self.model_id)
        except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover - depends on optional model download
            raise RuntimeError(
                "Failed to load human parsing model "
                f"'{self.model_id}'. Install model dependencies and verify network or local cache access."
            ) from exc

        self._image_processor = image_processor
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        raw_id_to_label = getattr(self._model.config, "id2label", {}) or {}
        normalized_config_labels = {
            int(label_id): _normalize_label_name(label_name)
            for label_id, label_name in raw_id_to_label.items()
        }

        if self.model_id == DEFAULT_MODEL_ID:
            if normalized_config_labels and normalized_config_labels != DEFAULT_MODEL_ID_TO_LABEL:
                raise RuntimeError(
                    "Loaded human parsing model labels do not match the verified label mapping "
                    f"for '{DEFAULT_MODEL_ID}'."
                )
            self.id_to_label = dict(DEFAULT_MODEL_ID_TO_LABEL)
            self.label_id_to_part = dict(DEFAULT_MODEL_LABEL_ID_TO_PART)
            return

        self.id_to_label = normalized_config_labels
        self.label_id_to_part = {
            label_id: DEFAULT_MODEL_LABEL_TO_PART[label_name]
            for label_id, label_name in self.id_to_label.items()
            if label_name in DEFAULT_MODEL_LABEL_TO_PART
        }

    def infer(self, image_bhwc: torch.Tensor) -> list[np.ndarray]:
        if image_bhwc.ndim != 4 or image_bhwc.shape[-1] != 3:
            raise ValueError("Expected image tensor with shape [B, H, W, 3].")

        self.load(image_bhwc.device)
        assert self._image_processor is not None
        assert self._model is not None
        assert self._device is not None

        try:
            from PIL import Image
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("CutoutRiggingSplitter requires Pillow for image conversion.") from exc

        image_uint8 = (
            image_bhwc.detach()
            .cpu()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )

        outputs: list[np.ndarray] = []
        for image in image_uint8:
            pil_image = Image.fromarray(image)
            inputs = self._image_processor(images=pil_image, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits

            resized_logits = F.interpolate(
                logits,
                size=pil_image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            labels = resized_logits.argmax(dim=1)[0].detach().cpu().to(torch.int32).numpy()
            outputs.append(labels)

        return outputs
