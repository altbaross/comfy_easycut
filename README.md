# comfy_easycut

Minimal ComfyUI custom node package for preparing full-canvas cutout rigging layers from a single person image.

## Included node

- `CutoutRiggingSplitter`

## Current scope

Current milestone level: stable full-canvas outputs through the initial feathering stage. Optional crop outputs, alternate parsing backends, and pose-assisted refinement are intentionally not implemented yet.

This implementation keeps the default output schema stable for ComfyUI:

- full-canvas RGB image per canonical part
- full-canvas mask per canonical part
- `limbs_union_mask`
- `torso_hole_mask`

The node uses a single lazy-loaded human parsing backend interface. The backend supports only the verified Hugging Face semantic segmentation model `mattmdjaga/segformer_b2_clothes`, and raises a clear runtime error if the optional dependency or model is unavailable.

`feathering_amount` softly feathers returned part masks and part images for display, while `padding` expands the torso region used for the conservative `torso_hole_mask`.

## Notes

- ComfyUI `IMAGE` tensors are expected as `[B, H, W, 3]` float32 in `[0, 1]`
- ComfyUI `MASK` tensors are returned as `[B, H, W]` float32 in `[0, 1]`
- Missing parts return zero images and zero masks
- Multiple-person scenes currently rely on the backend's scene-level parsing behavior
