from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


class RootPackageLoadingTests(unittest.TestCase):
    def test_root_package_loads_without_parent_package_in_sys_modules(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        init_path = repo_root / "__init__.py"
        module_name = "custom_nodes.comfy_easycut"

        previous_modules = {
            name: sys.modules.get(name)
            for name in ("custom_nodes", module_name)
        }
        previous_sys_path = list(sys.path)

        try:
            sys.modules.pop("custom_nodes", None)
            sys.modules.pop(module_name, None)
            sys.path = [path for path in sys.path if path != str(repo_root)]

            spec = importlib.util.spec_from_file_location(
                module_name,
                init_path,
                submodule_search_locations=[str(repo_root)],
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self.assertIn("CutoutRiggingSplitter", module.NODE_CLASS_MAPPINGS)
            self.assertEqual(
                module.NODE_DISPLAY_NAME_MAPPINGS["CutoutRiggingSplitter"],
                "Cutout Rigging Splitter",
            )
        finally:
            sys.path = previous_sys_path
            for name, previous in previous_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
