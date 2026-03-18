# comfy_easycut

Minimal ComfyUI custom node package for preparing cutout rigging layers from a single primary person image.

## Included node

- `CutoutRiggingSplitter`
- `GoogleNanoBananaConnector`

## Current scope

`CutoutRiggingSplitter` is implemented as a production-oriented ComfyUI node tuned for 2D animation illustration cutout prep, with a stable default output schema:

- full-canvas RGB image per canonical part
- full-canvas mask per canonical part
- `limbs_union_mask`
- `torso_hole_mask`

The node uses a single lazy-loaded human parsing backend interface. By default it uses the verified Hugging Face semantic segmentation model `mattmdjaga/segformer_b2_clothes`, and raises a clear runtime error if the dependency or model is unavailable.

### Canonical parts

The node groups backend labels into these canonical rigging parts:

- `head`
- `eyes`
- `hair`
- `torso`
- `arm_left`
- `arm_right`
- `leg_left`
- `leg_right`

The backend-specific mapping is explicit and verified. Hair maps to `hair`, face-derived eye windows map to `eyes`, dress/scarf/skirt/belt map to `torso`, left/right shoes map to the corresponding leg outputs, and the shared `pants` label is split across `leg_left` and `leg_right` by mask midpoint. The remaining face/head area stays in `head`, and missing parts return zero images and zero masks instead of raising errors.

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

Long-sleeve `upper-clothes` pixels that touch visible arm regions are also reassigned from `torso` into the corresponding arm mask before primary-person selection, which helps sleeves stay attached to arm outputs instead of being lost to the torso cut.

## Install

For a manual ComfyUI install, clone this repository into `ComfyUI/custom_nodes/comfy_easycut`, then install the runtime dependencies with the same Python environment that launches ComfyUI:

```bash
python -m pip install -r requirements.txt
```

If you are using a portable ComfyUI build on Windows, run the command with ComfyUI's embedded Python from the cloned node directory, for example:

```bash
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

After installing the dependencies, restart ComfyUI and confirm that `Cutout Rigging Splitter` appears in the custom node list.

## Optional Google Nano Banana / Gemini parsing backend

For Google multimodal parsing, either connect a `GoogleNanoBananaConnector` node into the `human_parsing_backend` input on `CutoutRiggingSplitter`, or set:

```bash
export COMFY_EASYCUT_PARSING_BACKEND=google_nano_banana
export GOOGLE_API_KEY=your_google_api_key
```

Optional overrides:

```bash
export COMFY_EASYCUT_GOOGLE_MODEL=gemini-2.5-flash-image
export COMFY_EASYCUT_GOOGLE_API_BASE=https://generativelanguage.googleapis.com/v1beta/models
export COMFY_EASYCUT_GOOGLE_TIMEOUT_SECONDS=60
```

The connector node exposes the same configuration directly in the ComfyUI graph:

- `api_key`
- `model_id`
- `api_base`
- `timeout_seconds`

The Google image API is used for image recognition and structured region extraction. Because Nano Banana/Gemini does not directly return dense semantic masks, this backend asks the model for strict JSON segment regions and reconstructs the label mask locally from returned row spans and boxes before passing it through the standard rigging pipeline.

## Notes

- ComfyUI `IMAGE` tensors are expected as `[B, H, W, 3]` float32 in `[0, 1]`
- ComfyUI `MASK` tensors are returned as `[B, H, W]` float32 in `[0, 1]`
- Full-canvas RGB + MASK outputs remain the default behavior
- Optional pose refinement is a no-op unless a compatible refiner backend is injected
