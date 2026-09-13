"""Load a module from examples/ by filename — examples/ is not a package
(it is meant to be read and copied, not imported as a library), so tests
that exercise it load each file directly rather than adding an __init__.py
that would change what pip installs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def load_example(filename: str):
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(f"examples.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
