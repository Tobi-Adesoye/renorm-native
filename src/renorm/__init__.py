"""
Renorm package root.
Clean export layer (no side effects, no circular imports).
Designed for safe pip install + CI + CPU/GPU environments.
"""

__version__ = "1.1.0"

import importlib

__all__ = [
    "RenormLinear",
    "RenormLinearFunction",
    "RenormTransformerLayer",
]


def __getattr__(name: str):
    """
    Lazy loader to prevent import-time failures in CI / CPU-only environments.
    """
    if name in __all__:
        module = importlib.import_module("renorm.layers")
        return getattr(module, name)

    raise AttributeError(f"module 'renorm' has no attribute '{name}'")


# ✅ ADD THIS AT THE VERY BOTTOM
def __dir__():
    return sorted(__all__)