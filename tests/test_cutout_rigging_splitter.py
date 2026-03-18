from __future__ import annotations

import unittest

import numpy as np
import torch

from comfyui_cutout_rigging_splitter import NODE_CLASS_MAPPINGS
from comfyui_cutout_rigging_splitter.backends.base import BaseHumanParsingBackend
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

    def load(self, device: torch.device) -> None:
        del device

    def infer(self, image_bhwc: torch.Tensor) -> list[np.ndarray]:
        self.load(image_bhwc.device)
        return self.outputs


class CutoutRiggingSplitterTests(unittest.TestCase):
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

        result = node.process(image, feathering_amount=2, padding=8)

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

        self.assertTrue(torch.equal(head_mask[0, 0], torch.tensor([1.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(torso_mask[0, 0], torch.tensor([0.0, 0.0, 1.0])))
        self.assertEqual(float(arm_left_mask[1].sum()), 0.0)
        self.assertEqual(float(arm_right_mask[1].sum()), 0.0)
        self.assertEqual(float(leg_left_mask[1].sum()), 0.0)
        self.assertEqual(float(leg_right_mask[1].sum()), 0.0)
        self.assertEqual(float(limbs_union_mask[1].sum()), 0.0)

        head_image = result[0]
        self.assertTrue(torch.equal(head_image[0], image[0] * head_mask[0].unsqueeze(-1)))


if __name__ == "__main__":
    unittest.main()
