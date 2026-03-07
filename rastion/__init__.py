"""Rastion routing-first solver hub package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rastion")
except PackageNotFoundError:
    __version__ = "0.1.2"

__all__ = ["__version__"]
