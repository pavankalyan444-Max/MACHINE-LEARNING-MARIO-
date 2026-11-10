import os
from collections.abc import Mapping
from typing import Any

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "trained_models"
)


def model_path(filename: str, create: bool = False) -> str:
    """Resolve a bare checkpoint filename inside MODEL_DIR."""
    if os.path.dirname(filename):
        path = filename
    else:
        path = os.path.join(MODEL_DIR, filename)
    if create:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def checkpoint_value(
    checkpoint: Mapping[str, Any],
    *keys: str
) -> Any:
    for key in keys:
        if key in checkpoint:
            return checkpoint[key]

    expected_keys = ", ".join(repr(key) for key in keys)
    raise KeyError(
        f"Checkpoint is missing required key; expected one of: {expected_keys}"
    )
