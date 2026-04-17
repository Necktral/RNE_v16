"""Alias legacy -> runtime.agents."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("runtime.agents")
_sys.modules[__name__] = _module

