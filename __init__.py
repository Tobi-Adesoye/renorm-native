import importlib.util

__version__ = "1.0.1"

# Optional availability check (no unused imports)
if importlib.util.find_spec("renorm") is None:
    raise ModuleNotFoundError("renorm package not found")

from renorm.layers import RenormLinear, RenormLinearFunction

__all__ = [
    "RenormLinear",
    "RenormLinearFunction",
]
