from importlib import import_module
from pathlib import Path
import sys


def _load_node_package():
    if __package__:
        try:
            return import_module(".comfyui_cutout_rigging_splitter", __package__)
        except ModuleNotFoundError as exc:
            allowed_missing = {
                "custom_nodes",
                f"{__package__}.comfyui_cutout_rigging_splitter",
            }
            if exc.name not in allowed_missing:
                raise

    package_dir = str(Path(__file__).resolve().parent)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    return import_module("comfyui_cutout_rigging_splitter")


_node_package = _load_node_package()
NODE_CLASS_MAPPINGS = _node_package.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _node_package.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
