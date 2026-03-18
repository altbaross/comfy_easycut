from __future__ import annotations

import unittest

import numpy as np
import torch

from comfyui_cutout_rigging_splitter import NODE_CLASS_MAPPINGS
from comfyui_cutout_rigging_splitter.backends.base import BaseHumanParsingBackend
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


class LoadTrackingBackend(StubParsingBackend):
    def __init__(self, outputs: list[np.ndarray]) -> None:
        super().__init__(outputs)
        self.loaded_device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        self.loaded_device = device

    def infer(self, image_bhwc: torch.Tensor) -> list[np.ndarray]:
        if self.loaded_device != image_bhwc.device:
            raise RuntimeError("backend load() must be called with the image device before infer().")
        return self.outputs


class CutoutRiggingSplitterTests(unittest.TestCase):
    def test_default_backend_uses_explicit_verified_label_constants(self) -> None:
        self.assertEqual(DEFAULT_MODEL_ID_TO_LABEL[11], "face")
        self.assertEqual(DEFAULT_MODEL_ID_TO_LABEL[14], "left-arm")
        self.assertEqual(DEFAULT_MODEL_ID_TO_LABEL[15], "right-arm")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[11], "head")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[14], "arm_left")
        self.assertEqual(DEFAULT_MODEL_LABEL_ID_TO_PART[15], "arm_right")

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

        self.assertEqual(len(result), 14)
        for index, tensor in enumerate(result):
            expected_shape = (2, 4, 3, 3) if index % 2 == 0 and index < 12 else (2, 4, 3)
            self.assertEqual(tensor.shape, expected_shape)
            self.assertEqual(tensor.dtype, torch.float32)

        head_mask = result[1]
        torso_mask = result[3]
        arm_left_mask = result[5]
        arm_right_mask = result[7]
        leg_left_mask = result[9]
        leg_right_mask = result[11]
        limbs_union_mask = result[12]
        torso_hole_mask = result[13]

        self.assertTrue(torch.equal(head_mask[0, 0], torch.tensor([1.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(torso_mask[0, 0], torch.tensor([0.0, 0.0, 1.0])))
        self.assertEqual(float(arm_left_mask[1].sum()), 0.0)
        self.assertEqual(float(arm_right_mask[1].sum()), 0.0)
        self.assertEqual(float(leg_left_mask[1].sum()), 0.0)
        self.assertEqual(float(leg_right_mask[1].sum()), 0.0)
        self.assertEqual(float(limbs_union_mask[1].sum()), 0.0)
        self.assertEqual(float(torso_hole_mask[0].sum()), 0.0)

        head_image = result[0]
        self.assertTrue(torch.equal(head_image[0], image[0] * head_mask[0].unsqueeze(-1)))

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

        result = node.process(image, feathering_amount=16, padding=128)
        self.assertEqual(len(result), 14)
        with self.assertRaisesRegex(ValueError, "feathering_amount"):
            node.process(image, feathering_amount=-1, padding=0)
        with self.assertRaisesRegex(ValueError, "padding"):
            node.process(image, feathering_amount=0, padding=129)

    def test_process_loads_backend_on_image_device_before_infer(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        backend = LoadTrackingBackend(outputs)
        node = CutoutRiggingSplitter(backend=backend)

        result = node.process(image, feathering_amount=0, padding=0)

        self.assertEqual(len(result), 14)
        self.assertEqual(backend.loaded_device, image.device)

    def test_process_requires_backend_label_mask_list(self) -> None:
        image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = [np.zeros((4, 4), dtype=np.int32)]
        node = CutoutRiggingSplitter(backend=NonListParsingBackend(outputs))

        with self.assertRaisesRegex(RuntimeError, "must return a list"):
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
