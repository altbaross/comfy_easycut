from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import numpy as np
import torch

from comfyui_cutout_rigging_splitter import NODE_CLASS_MAPPINGS
from comfyui_cutout_rigging_splitter.backends import build_human_parsing_backend_from_environment
from comfyui_cutout_rigging_splitter.backends.base import BaseHumanParsingBackend
from comfyui_cutout_rigging_splitter.backends.google_nano_banana_parsing import (
    GOOGLE_NANO_BANANA_BACKEND_NAME,
    GoogleNanoBananaParsingBackend,
)
from comfyui_cutout_rigging_splitter.backends.pose_backend_optional import BasePoseRefinementBackend
from comfyui_cutout_rigging_splitter.backends.transformers_human_parsing import (
    DEFAULT_MODEL_ID_TO_LABEL,
    DEFAULT_MODEL_LABEL_ID_TO_PART,
    TransformersHumanParsingBackend,
)
from comfyui_cutout_rigging_splitter.nodes import CutoutRiggingSplitter


class StubParsingBackend(BaseHumanParsingBackend):
    def __init__(self, outputs: list[np.ndarray]) -> None:
        super().__init__()
        self.outputs = outputs
        self.id_to_label = {
            0: "background",
            1: "face",
            2: "upper-clothes",
            3: "left-arm",
            4: "right-arm",
            5: "left-leg",
            6: "right-leg",
        }
        self.label_id_to_part = {
            1: "head",
            2: "torso",
            3: "arm_left",
            4: "arm_right",
            5: "leg_left",
            6: "leg_right",
        }

    def load(self, device: torch.device) -> None:
        del device

    def infer(self, image_bhwc: torch.Tensor) -> list[np.ndarray]:
        self.load(image_bhwc.device)
        return self.outputs


class NonListParsingBackend(StubParsingBackend):
    def infer(self, image_bhwc: torch.Tensor) -> object:
        self.load(image_bhwc.device)
        return tuple(self.outputs)


class MissingMappingBackend(StubParsingBackend):
    def __init__(self, outputs: list[np.ndarray]) -> None:
        super().__init__(outputs)
        del self.label_id_to_part


class MissingLoadBackend(StubParsingBackend):
    def __getattribute__(self, name: str) -> object:
        if name == "load":
            raise AttributeError(name)
        return super().__getattribute__(name)


class NonCallableLoadBackend(StubParsingBackend):
    def __init__(self, outputs: list[np.ndarray]) -> None:
        super().__init__(outputs)
        self.load = None


class MissingInferBackend(StubParsingBackend):
    def __getattribute__(self, name: str) -> object:
        if name == "infer":
            raise AttributeError(name)
        return super().__getattribute__(name)


class NonCallableInferBackend(StubParsingBackend):
    def __init__(self, outputs: list[np.ndarray]) -> None:
        super().__init__(outputs)
        self.infer = None


class LoadTrackingBackend(StubParsingBackend):
    def __init__(self, outputs: list[np.ndarray]) -> None:
        super().__init__(outputs)
        self.loaded_device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        if not isinstance(device, torch.device):
            raise RuntimeError("load() must receive a torch.device instance.")
        self.loaded_device = device

    def infer(self, image_bhwc: torch.Tensor) -> list[np.ndarray]:
        if self.loaded_device != image_bhwc.device:
            raise RuntimeError("backend load() must be called with the image device before infer().")
        return self.outputs


class ClothingParsingBackend(BaseHumanParsingBackend):
    def __init__(self, outputs: list[np.ndarray]) -> None:
        super().__init__()
        self.outputs = outputs
        self.id_to_label = dict(DEFAULT_MODEL_ID_TO_LABEL)
        self.label_id_to_part = dict(DEFAULT_MODEL_LABEL_ID_TO_PART)

    def load(self, device: torch.device) -> None:
        del device

    def infer(self, image_bhwc: torch.Tensor) -> list[np.ndarray]:
        self.load(image_bhwc.device)
        return self.outputs


class TrackingPoseRefiner(BasePoseRefinementBackend):
    def __init__(self) -> None:
        self.loaded_device: torch.device | None = None
        self.called = False

    def load(self, device: torch.device) -> None:
        self.loaded_device = device

    def refine(
        self,
        image_bhwc: torch.Tensor,
        part_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del image_bhwc
        self.called = True
        refined = dict(part_masks)
        refined["head"] = torch.zeros_like(refined["head"])
        return refined


class GoogleNanoBananaRequestRecorder:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def __call__(self, url: str, headers: dict[str, str], data: bytes, timeout: float) -> str:
        self.calls.append((url, headers, data, timeout))
        return self.response_text


class CutoutRiggingSplitterTests(unittest.TestCase):
    def test_google_nano_banana_backend_requires_api_key(self) -> None:
        backend = GoogleNanoBananaParsingBackend(api_key="")

        with self.assertRaisesRegex(RuntimeError, "API key"):
            backend.load(torch.device("cpu"))

    def test_google_nano_banana_backend_builds_gemini_request_and_rasterizes_segments(self) -> None:
        recorder = GoogleNanoBananaRequestRecorder(
            json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "analysis": {
                                                    "subject_summary": "single anime character",
                                                    "visual_style": "2d_animation_illustration",
                                                    "primary_character_count": 1,
                                                },
                                                "segments": [
                                                    {"label_id": 2, "label": "hair", "rows": [{"y": 0, "x0": 1, "x1": 3}]},
                                                    {"label_id": 11, "label": "face", "boxes": [{"y0": 1, "x0": 1, "y1": 3, "x1": 3}]},
                                                    {"label": "left-arm", "rows": [{"y": 2, "x0": 0, "x1": 1}]},
                                                ],
                                            }
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        )
        backend = GoogleNanoBananaParsingBackend(
            api_key="test-key",
            model_id="gemini-test",
            request_sender=recorder,
        )
        image = torch.ones((1, 3, 4, 3), dtype=torch.float32)

        outputs = backend.infer(image)

        self.assertEqual(len(outputs), 1)
        mask = outputs[0]
        self.assertEqual(mask.dtype, np.int32)
        self.assertEqual(int(mask[0, 1]), 2)
        self.assertEqual(int(mask[1, 1]), 11)
        self.assertEqual(int(mask[2, 0]), 14)
        self.assertEqual(backend.last_analysis[0]["visual_style"], "2d_animation_illustration")
        self.assertEqual(len(recorder.calls), 1)
        request_url, headers, data, timeout = recorder.calls[0]
        self.assertIn("gemini-test:generateContent", request_url)
        self.assertIn("key=test-key", request_url)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(timeout, 60.0)
        request_body = json.loads(data.decode("utf-8"))
        self.assertEqual(request_body["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(request_body["contents"][0]["parts"][1]["inline_data"]["mime_type"], "image/png")

    def test_google_nano_banana_backend_accepts_direct_json_payload(self) -> None:
        backend = GoogleNanoBananaParsingBackend(
            api_key="test-key",
            request_sender=GoogleNanoBananaRequestRecorder(
                json.dumps(
                    {
                        "analysis": {"subject_summary": "direct payload"},
                        "segments": [
                            {"label": "pants", "boxes": [{"y0": 0, "x0": 0, "y1": 2, "x1": 2}]},
                        ],
                    }
                )
            ),
        )
        image = torch.ones((1, 2, 3, 3), dtype=torch.float32)

        outputs = backend.infer(image)

        self.assertEqual(int(outputs[0][0, 0]), 6)
        self.assertEqual(backend.last_analysis[0]["subject_summary"], "direct payload")

    def test_build_human_parsing_backend_from_environment_supports_google_nano_banana(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "COMFY_EASYCUT_PARSING_BACKEND": GOOGLE_NANO_BANANA_BACKEND_NAME,
                "GOOGLE_API_KEY": "env-key",
                "COMFY_EASYCUT_GOOGLE_MODEL": "gemini-env",
            },
            clear=False,
        ):
            backend = build_human_parsing_backend_from_environment()

        self.assertIsInstance(backend, GoogleNanoBananaParsingBackend)
        self.assertEqual(backend.api_key, "env-key")
        self.assertEqual(backend.model_id, "gemini-env")

    def test_build_human_parsing_backend_from_environment_rejects_unknown_backend(self) -> None:
        with mock.patch.dict(os.environ, {"COMFY_EASYCUT_PARSING_BACKEND": "unknown-backend"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "Unsupported COMFY_EASYCUT_PARSING_BACKEND"):
                build_human_parsing_backend_from_environment()

    def test_process_returns_zeros_when_no_parts_detected(self) -> None:
        image = torch.ones((1, 3, 2, 3), dtype=torch.float32)
        outputs = [np.zeros((3, 2), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=8)

        self.assertEqual(len(result), 18)
        for index, tensor in enumerate(result):
            self.assertEqual(float(tensor.sum()), 0.0, f"expected zero output at index {index}")

    def test_default_backend_uses_explicit_verified_label_constants(self) -> None:
        self.assertEqual(DEFAULT_MODEL_ID_TO_LABEL[11], "face")
        self.assertEqual(DEFAULT_MODEL_ID_TO_LABEL[14], "left-arm")
        self.assertEqual(DEFAULT_MODEL_ID_TO_LABEL[15], "right-arm")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[5], "torso")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[6], "pants")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[8], "torso")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[2], "hair")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[11], "head")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[14], "arm_left")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[15], "arm_right")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[9], "leg_left")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[10], "leg_right")

    def test_node_registration_exports_expected_class(self) -> None:
        self.assertIn("CutoutRiggingSplitter", NODE_CLASS_MAPPINGS)
        self.assertIs(NODE_CLASS_MAPPINGS["CutoutRiggingSplitter"], CutoutRiggingSplitter)

    def test_process_returns_full_canvas_outputs_for_batch(self) -> None:
        image = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3) / 100.0
        outputs = [
            np.array(
                [
                    [1, 1, 2],
                    [3, 2, 2],
                    [5, 0, 6],
                    [5, 0, 6],
                ],
                dtype=np.int32,
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int32,
            ),
        ]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0)

        self.assertEqual(len(result), 18)
        for index, tensor in enumerate(result):
            expected_shape = (2, 4, 3, 3) if index % 2 == 0 and index < 16 else (2, 4, 3)
            self.assertEqual(tensor.shape, expected_shape)
            self.assertEqual(tensor.dtype, torch.float32)

        head_mask = result[1]
        eyes_mask = result[3]
        hair_mask = result[5]
        torso_mask = result[7]
        arm_left_mask = result[9]
        arm_right_mask = result[11]
        leg_left_mask = result[13]
        leg_right_mask = result[15]
        limbs_union_mask = result[16]

        self.assertTrue(torch.equal(head_mask[0, 0], torch.tensor([1.0, 1.0, 0.0])))
        self.assertEqual(float(eyes_mask.sum()), 0.0)
        self.assertEqual(float(hair_mask.sum()), 0.0)
        self.assertTrue(torch.equal(torso_mask[0, 0], torch.tensor([0.0, 0.0, 1.0])))
        self.assertEqual(float(arm_left_mask[1].sum()), 0.0)
        self.assertEqual(float(arm_right_mask[1].sum()), 0.0)
        self.assertEqual(float(leg_left_mask[1].sum()), 0.0)
        self.assertEqual(float(leg_right_mask[1].sum()), 0.0)
        self.assertEqual(float(limbs_union_mask[1].sum()), 0.0)
        head_image = result[0]
        self.assertTrue(torch.equal(head_image[0], image[0] * head_mask[0].unsqueeze(-1)))

    def test_crop_mode_returns_per_part_crops_for_single_image(self) -> None:
        image = torch.arange(1 * 6 * 6 * 3, dtype=torch.float32).reshape(1, 6, 6, 3) / 255.0
        outputs = [
            np.array(
                [
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 3, 2, 2, 0, 0],
                    [0, 0, 2, 2, 0, 0],
                    [5, 0, 0, 0, 6, 0],
                    [5, 0, 0, 0, 6, 0],
                ],
                dtype=np.int32,
            )
        ]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0, crop_mode=True, crop_padding=0)

        self.assertEqual(result[0].shape, (1, 2, 2, 3))
        self.assertEqual(result[1].shape, (1, 2, 2))
        self.assertEqual(result[6].shape, (1, 2, 2, 3))
        self.assertEqual(result[11].shape, (1, 1, 1))
        self.assertEqual(float(result[11].sum()), 0.0)
        self.assertEqual(node.last_crop_boxes["head"], (0, 2, 0, 2))
        self.assertIsNone(node.last_crop_boxes["eyes"])
        self.assertIsNone(node.last_crop_boxes["hair"])
        self.assertIsNone(node.last_crop_boxes["arm_right"])

    def test_crop_mode_falls_back_to_full_canvas_for_batch(self) -> None:
        image = torch.ones((2, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32), np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0, crop_mode=True, crop_padding=0)

        self.assertEqual(result[0].shape, (2, 4, 4, 3))
        self.assertEqual(result[1].shape, (2, 4, 4))
        self.assertEqual(node.last_crop_boxes, {})

    def test_primary_person_selection_keeps_largest_visible_component(self) -> None:
        image = torch.ones((1, 6, 6, 3), dtype=torch.float32)
        outputs = [
            np.array(
                [
                    [0, 1, 1, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 2, 0],
                ],
                dtype=np.int32,
            )
        ]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0)

        head_mask = result[1][0]
        torso_mask = result[7][0]
        self.assertEqual(float(head_mask[4, 4]), 0.0)
        self.assertEqual(float(torso_mask[5, 4]), 0.0)
        self.assertEqual(float(head_mask[0, 1]), 1.0)
        self.assertGreater(float(torso_mask.sum()), 0.0)

    def test_process_splits_pants_label_between_left_and_right_legs_for_each_sample(self) -> None:
        image = torch.ones((2, 4, 6, 3), dtype=torch.float32)
        outputs = [
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 6, 6, 6, 6, 0],
                    [0, 6, 6, 6, 6, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [6, 6, 6, 0, 0, 0],
                    [6, 6, 6, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
        ]
        node = CutoutRiggingSplitter(backend=ClothingParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0)

        leg_left_mask = result[13]
        leg_right_mask = result[15]
        self.assertTrue(torch.equal(leg_left_mask[0, 1], torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(leg_right_mask[0, 1], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(leg_left_mask[1, 1], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(leg_right_mask[1, 1], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])))

    def test_process_maps_skirt_and_belt_labels_to_torso(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 5, 8, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
        ]
        node = CutoutRiggingSplitter(backend=ClothingParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0)

        torso_mask = result[7][0]
        self.assertEqual(float(torso_mask[1, 1]), 1.0)
        self.assertEqual(float(torso_mask[1, 2]), 1.0)

    def test_process_separates_hair_and_derives_eyes_for_illustration_face_labels(self) -> None:
        image = torch.ones((1, 6, 8, 3), dtype=torch.float32)
        outputs = [
            np.array(
                [
                    [0, 0, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 2, 0, 0],
                    [0, 0, 11, 11, 11, 11, 0, 0],
                    [0, 0, 11, 11, 11, 11, 0, 0],
                    [0, 0, 11, 11, 11, 11, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
        ]
        node = CutoutRiggingSplitter(backend=ClothingParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0)

        head_mask = result[1][0]
        eyes_mask = result[3][0]
        hair_mask = result[5][0]
        self.assertEqual(float(hair_mask[0:2, 2:6].sum()), 8.0)
        self.assertEqual(float(hair_mask.sum()), 8.0)
        self.assertGreater(float(eyes_mask.sum()), 0.0)
        self.assertEqual(float(eyes_mask[:3].sum()), 0.0)
        # With a 4-pixel-wide face band, the heuristic keeps a one-pixel center
        # gap, so the synthetic illustration case produces exactly 3 eye pixels.
        self.assertEqual(float(eyes_mask[3, 2:6].sum()), 3.0)
        self.assertEqual(float(eyes_mask[:, :2].sum()), 0.0)
        self.assertEqual(float(eyes_mask[:, 6:].sum()), 0.0)
        self.assertEqual(float(eyes_mask[4:].sum()), 0.0)
        self.assertEqual(float(head_mask[0:2, 2:6].sum()), 0.0)
        self.assertEqual(float((head_mask * eyes_mask).sum()), 0.0)
        self.assertEqual(float(head_mask[4, 3]), 1.0)

    def test_process_redistributes_upper_clothes_touching_arm_to_arm_mask(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 14, 4, 0],
                    [0, 0, 4, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
        ]
        node = CutoutRiggingSplitter(backend=ClothingParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0)

        arm_left_mask = result[9][0]
        torso_mask = result[7][0]
        self.assertEqual(float(arm_left_mask.sum()), 3.0)
        self.assertEqual(float(arm_left_mask[1, 2]), 1.0)
        self.assertEqual(float(arm_left_mask[2, 2]), 1.0)
        self.assertEqual(float(torso_mask.sum()), 0.0)

    def test_torso_hole_mask_is_conservative_overlap_near_torso(self) -> None:
        image = torch.ones((1, 6, 6, 3), dtype=torch.float32)
        outputs = [
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [3, 3, 2, 2, 0, 0],
                    [0, 0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
        ]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        result = node.process(image, feathering_amount=0, padding=0)

        limbs_union_mask = result[16][0]
        torso_hole_mask = result[17][0]
        self.assertEqual(float(limbs_union_mask.sum()), 2.0)
        self.assertEqual(float(torso_hole_mask.sum()), 1.0)
        self.assertEqual(float(torso_hole_mask[2, 1]), 1.0)
        self.assertEqual(float(torso_hole_mask[2, 0]), 0.0)

    def test_feathering_softens_output_masks_without_breaking_shapes(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
        ]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        result = node.process(image, feathering_amount=1, padding=1)

        head_mask = result[1]
        self.assertEqual(head_mask.shape, (1, 4, 4))
        self.assertGreater(float(head_mask[0, 1, 0]), 0.0)
        self.assertLess(float(head_mask[0, 1, 0]), 1.0)

    def test_pose_refinement_hook_is_optional_and_applied_when_enabled(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.ones((4, 4), dtype=np.int32)]
        pose_refiner = TrackingPoseRefiner()
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs), pose_refiner=pose_refiner)

        result = node.process(image, feathering_amount=0, padding=0, enable_pose_refinement=True)

        self.assertTrue(pose_refiner.called)
        self.assertEqual(pose_refiner.loaded_device, image.device)
        self.assertEqual(float(result[1].sum()), 0.0)

    def test_process_raises_clear_error_for_invalid_backend_mask_shape(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((2, 2), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        with self.assertRaisesRegex(RuntimeError, "invalid label mask shape"):
            node.process(image, feathering_amount=0, padding=0)

    def test_process_validates_parameter_ranges(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=StubParsingBackend(outputs))

        result = node.process(
            image,
            feathering_amount=16,
            padding=128,
            crop_padding=128,
            morphology_strength=8,
            mask_threshold=1.0,
        )
        self.assertEqual(len(result), 18)
        with self.assertRaisesRegex(ValueError, "feathering_amount"):
            node.process(image, feathering_amount=-1, padding=0)
        with self.assertRaisesRegex(ValueError, "padding"):
            node.process(image, feathering_amount=0, padding=129)
        with self.assertRaisesRegex(ValueError, "crop_padding"):
            node.process(image, feathering_amount=0, padding=0, crop_padding=129)
        with self.assertRaisesRegex(ValueError, "morphology_strength"):
            node.process(image, feathering_amount=0, padding=0, morphology_strength=9)
        with self.assertRaisesRegex(ValueError, "mask_threshold"):
            node.process(image, feathering_amount=0, padding=0, mask_threshold=1.1)

    def test_process_loads_backend_on_image_device_before_infer(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        backend = LoadTrackingBackend(outputs)
        node = CutoutRiggingSplitter(backend=backend)

        result = node.process(image, feathering_amount=0, padding=0)

        self.assertEqual(len(result), 18)
        self.assertEqual(backend.loaded_device, image.device)

    def test_process_requires_backend_load_method(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=MissingLoadBackend(outputs))

        with self.assertRaisesRegex(RuntimeError, "load\\(device: torch\\.device\\)"):
            node.process(image, feathering_amount=0, padding=0)

    def test_process_requires_callable_backend_load_method(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=NonCallableLoadBackend(outputs))

        with self.assertRaisesRegex(RuntimeError, "callable"):
            node.process(image, feathering_amount=0, padding=0)

    def test_process_requires_backend_label_mask_list(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=NonListParsingBackend(outputs))

        with self.assertRaisesRegex(RuntimeError, "must return a list"):
            node.process(image, feathering_amount=0, padding=0)

    def test_process_requires_backend_infer_method(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=MissingInferBackend(outputs))

        with self.assertRaisesRegex(RuntimeError, "infer\\(image_bhwc: torch\\.Tensor\\)"):
            node.process(image, feathering_amount=0, padding=0)

    def test_process_requires_callable_backend_infer_method(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=NonCallableInferBackend(outputs))

        with self.assertRaisesRegex(RuntimeError, "infer attribute must be callable"):
            node.process(image, feathering_amount=0, padding=0)

    def test_process_requires_backend_label_mapping(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=MissingMappingBackend(outputs))

        with self.assertRaisesRegex(RuntimeError, "label_id_to_part"):
            node.process(image, feathering_amount=0, padding=0)

    def test_backend_rejects_unverified_model_id(self) -> None:
        backend = TransformersHumanParsingBackend(model_id="custom/model")

        with self.assertRaisesRegex(RuntimeError, "currently supports only the verified human parsing model"):
            backend.load(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
