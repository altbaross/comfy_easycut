from __future__ import annotations

import base64
import io
import json
import os
from typing import Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
import torch

from .base import BaseHumanParsingBackend
from .transformers_human_parsing import DEFAULT_MODEL_ID_TO_LABEL, DEFAULT_MODEL_LABEL_ID_TO_PART


GOOGLE_NANO_BANANA_BACKEND_NAME = "google_nano_banana"
GOOGLE_NANO_BANANA_MODEL_ID = "gemini-2.5-flash-image"
GOOGLE_NANO_BANANA_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GOOGLE_NANO_BANANA_TIMEOUT_SECONDS = 60.0

_REVERSE_LABEL_LOOKUP = {
    str(label_name).strip().lower().replace("_", "-").replace(" ", "-"): int(label_id)
    for label_id, label_name in DEFAULT_MODEL_ID_TO_LABEL.items()
}


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


class GoogleNanoBananaParsingBackend(BaseHumanParsingBackend):
    """Gemini/Nano Banana-backed parsing backend with local mask reconstruction.

    The Google image models are strong at multimodal image understanding but do
    not directly return dense semantic masks. This backend requests strict JSON
    region descriptions from the model and rasterizes those regions into the
    label-mask format required by ``CutoutRiggingSplitter``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = GOOGLE_NANO_BANANA_MODEL_ID,
        api_base: str = GOOGLE_NANO_BANANA_API_BASE,
        timeout_seconds: float = GOOGLE_NANO_BANANA_TIMEOUT_SECONDS,
        request_sender: Callable[[str, dict[str, str], bytes, float], str] | None = None,
    ) -> None:
        super().__init__()
        self.api_key = (api_key or "").strip()
        self.model_id = model_id
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.request_sender = request_sender or self._default_request_sender
        self.id_to_label = dict(DEFAULT_MODEL_ID_TO_LABEL)
        self.label_id_to_part = dict(DEFAULT_MODEL_LABEL_ID_TO_PART)
        self.last_analysis: list[dict[str, object]] = []
        self._loaded = False

    @classmethod
    def from_environment(cls) -> "GoogleNanoBananaParsingBackend":
        api_key = os.getenv("COMFY_EASYCUT_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        model_id = os.getenv("COMFY_EASYCUT_GOOGLE_MODEL", GOOGLE_NANO_BANANA_MODEL_ID)
        api_base = os.getenv("COMFY_EASYCUT_GOOGLE_API_BASE", GOOGLE_NANO_BANANA_API_BASE)
        timeout_seconds = float(os.getenv("COMFY_EASYCUT_GOOGLE_TIMEOUT_SECONDS", str(GOOGLE_NANO_BANANA_TIMEOUT_SECONDS)))
        return cls(
            api_key=api_key,
            model_id=model_id,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
        )

    def load(self, device: torch.device) -> None:
        del device
        if self._loaded:
            return
        if not self.api_key:
            raise RuntimeError(
                "Google Nano Banana parsing backend requires an API key. "
                "Set COMFY_EASYCUT_GOOGLE_API_KEY or GOOGLE_API_KEY."
            )
        self._loaded = True

    def infer(self, image_bhwc: torch.Tensor) -> list[np.ndarray]:
        if image_bhwc.ndim != 4 or image_bhwc.shape[-1] != 3:
            raise ValueError("Expected image tensor with shape [B, H, W, 3].")

        self.load(image_bhwc.device)
        self.last_analysis = []

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
            png_bytes = self._encode_png_bytes(image)
            raw_response = self._generate_content(png_bytes)
            parsed_payload = self._extract_model_payload(raw_response)
            self.last_analysis.append(self._coerce_analysis(parsed_payload))
            outputs.append(self._segments_to_mask(parsed_payload, height=image.shape[0], width=image.shape[1]))
        return outputs

    def _generate_content(self, png_bytes: bytes) -> str:
        request_url = (
            f"{self.api_base}/{urllib_parse.quote(self.model_id, safe='')}:generateContent"
            f"?key={urllib_parse.quote(self.api_key, safe='')}"
        )
        prompt = self._build_prompt()
        request_body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(png_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        return self.request_sender(
            request_url,
            {"Content-Type": "application/json"},
            json.dumps(request_body).encode("utf-8"),
            self.timeout_seconds,
        )

    @staticmethod
    def _default_request_sender(url: str, headers: dict[str, str], data: bytes, timeout: float) -> str:
        request = urllib_request.Request(url=url, headers=headers, data=data, method="POST")
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:  # pragma: no cover - depends on remote API
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Google Nano Banana API request failed with status {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:  # pragma: no cover - depends on network
            raise RuntimeError(f"Failed to reach Google Nano Banana API: {exc.reason}") from exc

    @staticmethod
    def _encode_png_bytes(image_hwc: np.ndarray) -> bytes:
        try:
            from PIL import Image
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("Google Nano Banana parsing backend requires Pillow for image conversion.") from exc

        buffer = io.BytesIO()
        Image.fromarray(image_hwc).save(buffer, format="PNG")
        return buffer.getvalue()

    def _build_prompt(self) -> str:
        label_lines = "\n".join(
            f"- {label_id}: {label_name}" for label_id, label_name in sorted(self.id_to_label.items()) if label_id != 0
        )
        return (
            "Analyze this single character image for 2D animation illustration cutout rigging.\n"
            "Return strict JSON only, with no markdown fences.\n"
            "Use this schema:\n"
            "{"
            '"analysis": {"subject_summary": "string value", "visual_style": "string value", "primary_character_count": 1}, '
            '"segments": ['
            '{"label_id": 11, "label": "face", "rows": [{"y": 0, "x0": 0, "x1": 1}], '
            '"boxes": [{"y0": 0, "x0": 0, "y1": 1, "x1": 1}]}'
            "]}"
            "\n"
            "Prefer dense row spans in 'rows' for tight segmentation. Use 'boxes' only when row spans are impractical.\n"
            "Bounds are integer pixel coordinates with x1/y1 as exclusive end coordinates.\n"
            "Only emit these labels:\n"
            f"{label_lines}\n"
            "Skip anything uncertain instead of inventing labels."
        )

    def _extract_model_payload(self, raw_response: str) -> dict[str, object]:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict) and isinstance(parsed.get("segments"), list):
            return parsed

        candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []
        text_parts: list[str] = []
        for candidate in candidates:
            content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
        if not text_parts:
            raise RuntimeError("Google Nano Banana API response did not include a JSON text payload.")

        candidate_text = _strip_json_fences("\n".join(text_parts))
        payload = json.loads(candidate_text)
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
            raise RuntimeError("Google Nano Banana API response JSON must contain a top-level 'segments' list.")
        return payload

    @staticmethod
    def _coerce_analysis(payload: dict[str, object]) -> dict[str, object]:
        analysis = payload.get("analysis", {})
        return analysis if isinstance(analysis, dict) else {}

    def _segments_to_mask(self, payload: dict[str, object], *, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.int32)
        segments = payload.get("segments", [])
        if not isinstance(segments, list):
            raise RuntimeError("Google Nano Banana API response JSON must contain a list of segments.")

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            label_id = self._resolve_label_id(segment)
            if label_id is None or label_id == 0:
                continue
            self._apply_rows(mask, label_id, segment.get("rows", []))
            self._apply_boxes(mask, label_id, segment.get("boxes", []))
        return mask

    def _resolve_label_id(self, segment: dict[str, object]) -> int | None:
        raw_label_id = segment.get("label_id")
        if raw_label_id is not None:
            try:
                label_id = int(raw_label_id)
            except (TypeError, ValueError):
                label_id = None
            else:
                return label_id if label_id in self.id_to_label else None

        raw_label = segment.get("label")
        normalized_label = str(raw_label).strip().lower().replace("_", "-").replace(" ", "-")
        return _REVERSE_LABEL_LOOKUP.get(normalized_label)

    @staticmethod
    def _apply_rows(mask: np.ndarray, label_id: int, rows: object) -> None:
        if not isinstance(rows, list):
            return
        height, width = mask.shape
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                y = int(row["y"])
                x0 = int(row["x0"])
                x1 = int(row["x1"])
            except (KeyError, TypeError, ValueError):
                continue
            if y < 0 or y >= height:
                continue
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            if x1 <= x0:
                continue
            mask[y, x0:x1] = label_id

    @staticmethod
    def _apply_boxes(mask: np.ndarray, label_id: int, boxes: object) -> None:
        if not isinstance(boxes, list):
            return
        height, width = mask.shape
        for box in boxes:
            if not isinstance(box, dict):
                continue
            try:
                y0 = int(box["y0"])
                x0 = int(box["x0"])
                y1 = int(box["y1"])
                x1 = int(box["x1"])
            except (KeyError, TypeError, ValueError):
                continue
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            if y1 <= y0 or x1 <= x0:
                continue
            mask[y0:y1, x0:x1] = label_id
