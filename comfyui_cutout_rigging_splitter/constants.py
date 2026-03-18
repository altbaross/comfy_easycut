CANONICAL_PARTS = (
    "head",
    "eyes",
    "hair",
    "torso",
    "arm_left",
    "arm_right",
    "leg_left",
    "leg_right",
)

MAX_FEATHERING_AMOUNT = 16
MAX_PADDING = 128
MAX_MORPHOLOGY_STRENGTH = 8
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_EMPTY_CROP_SIZE = 1

PART_OUTPUT_NAMES = (
    "head_image",
    "head_mask",
    "eyes_image",
    "eyes_mask",
    "hair_image",
    "hair_mask",
    "torso_image",
    "torso_mask",
    "arm_left_image",
    "arm_left_mask",
    "arm_right_image",
    "arm_right_mask",
    "leg_left_image",
    "leg_left_mask",
    "leg_right_image",
    "leg_right_mask",
    "limbs_union_mask",
    "torso_hole_mask",
)

RETURN_TYPES = (
    "IMAGE",
    "MASK",
    "IMAGE",
    "MASK",
    "IMAGE",
    "MASK",
    "IMAGE",
    "MASK",
    "IMAGE",
    "MASK",
    "IMAGE",
    "MASK",
    "IMAGE",
    "MASK",
    "IMAGE",
    "MASK",
    "MASK",
    "MASK",
)

RETURN_NAMES = PART_OUTPUT_NAMES
