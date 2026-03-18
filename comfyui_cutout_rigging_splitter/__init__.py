from .nodes import CutoutRiggingSplitter

NODE_CLASS_MAPPINGS = {
    "CutoutRiggingSplitter": CutoutRiggingSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutoutRiggingSplitter": "Cutout Rigging Splitter",
}

__all__ = [
    "CutoutRiggingSplitter",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
