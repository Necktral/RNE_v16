"""Alias legacy -> runtime.core."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("runtime.core")
_sys.modules[__name__] = _module

