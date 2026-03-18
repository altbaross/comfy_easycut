from .nodes import CutoutRiggingSplitter, GoogleNanoBananaConnector

NODE_CLASS_MAPPINGS = {
    "CutoutRiggingSplitter": CutoutRiggingSplitter,
    "GoogleNanoBananaConnector": GoogleNanoBananaConnector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutoutRiggingSplitter": "Cutout Rigging Splitter",
    "GoogleNanoBananaConnector": "Google Nano Banana Connector",
}

__all__ = [
    "CutoutRiggingSplitter",
    "GoogleNanoBananaConnector",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
