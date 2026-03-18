# comfy_easycut

Minimal ComfyUI custom node package for preparing cutout rigging layers from a single primary person image.

## Included node

- `CutoutRiggingSplitter`

## Current scope

`CutoutRiggingSplitter` is implemented as a production-oriented ComfyUI node with a stable default output schema:

- full-canvas RGB image per canonical part
- full-canvas mask per canonical part
- `limbs_union_mask`
- `torso_hole_mask`

The node uses a single lazy-loaded human parsing backend interface. The default backend supports only the verified Hugging Face semantic segmentation model `mattmdjaga/segformer_b2_clothes`, and raises a clear runtime error if the dependency or model is unavailable.

### Canonical parts

The node groups backend labels into these canonical rigging parts:

- `head`
- `torso`
- `arm_left`
- `arm_right`
- `leg_left`
- `leg_right`

The backend-specific mapping is explicit and verified. Hair maps to `head`, dress/scarf map to `torso`, and left/right shoes map to the corresponding leg outputs. Missing parts return zero images and zero masks instead of raising errors.

### Optional controls

Default processing is batch-safe and returns full-canvas outputs for `B=1` and `B>1`.

Optional controls add refinement without changing the default schema:

- `feathering_amount` softens returned display masks
- `morphology_strength` applies conservative logical-mask cleanup
- `padding` expands torso proximity when building `torso_hole_mask`
- `crop_mode` enables per-output cropped tensors for batch size `1`
- `crop_padding` expands crop boxes
- `enable_pose_refinement` activates an integration hook when a pose refiner backend is supplied

If `crop_mode` is enabled for `B>1`, the node safely falls back to full-canvas outputs. If a requested crop has no visible pixels, the node returns a small zero dummy crop for that output.

### Multiple people

When the parser produces multiple disconnected human regions, the node keeps only the largest connected canonical-part component so the outputs stay focused on the primary visible subject.

## Install

Install the runtime dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Notes

- ComfyUI `IMAGE` tensors are expected as `[B, H, W, 3]` float32 in `[0, 1]`
- ComfyUI `MASK` tensors are returned as `[B, H, W]` float32 in `[0, 1]`
- Full-canvas RGB + MASK outputs remain the default behavior
- Optional pose refinement is a no-op unless a compatible refiner backend is injected
