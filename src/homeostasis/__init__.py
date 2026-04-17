"""Alias src.homeostasis -> runtime.control.homeostasis."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("runtime.control.homeostasis")
_sys.modules[__name__] = _module

