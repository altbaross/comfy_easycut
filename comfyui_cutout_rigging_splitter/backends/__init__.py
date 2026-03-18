import os

from .base import BaseHumanParsingBackend
from .google_nano_banana_parsing import GOOGLE_NANO_BANANA_BACKEND_NAME, GoogleNanoBananaParsingBackend
from .pose_backend_optional import BasePoseRefinementBackend, NoOpPoseRefinementBackend
from .transformers_human_parsing import TransformersHumanParsingBackend


def build_human_parsing_backend_from_environment() -> BaseHumanParsingBackend:
    backend_name = os.getenv("COMFY_EASYCUT_PARSING_BACKEND", "transformers").strip().lower()
    if backend_name in {"", "transformers", "huggingface", "hf"}:
        return TransformersHumanParsingBackend()
    if backend_name in {GOOGLE_NANO_BANANA_BACKEND_NAME, "google", "nano_banana", "gemini"}:
        return GoogleNanoBananaParsingBackend.from_environment()
    raise RuntimeError(
        "Unsupported COMFY_EASYCUT_PARSING_BACKEND value "
        f"'{backend_name}'. Expected 'transformers' or '{GOOGLE_NANO_BANANA_BACKEND_NAME}'."
    )

__all__ = [
    "BaseHumanParsingBackend",
    "GoogleNanoBananaParsingBackend",
    "BasePoseRefinementBackend",
    "NoOpPoseRefinementBackend",
    "TransformersHumanParsingBackend",
    "build_human_parsing_backend_from_environment",
]
