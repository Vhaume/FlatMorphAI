# -*- coding: utf-8 -*-
"""regen_pipeline: inference chain of the planarian regeneration state, version v20 (20 October 2026).

Import order matters: _env installs cadrage_squelette_skan2 as cadrage_squelette and checks the GPU before any
production module is used. See README.md of the export for the description of every step.
"""
from . import _env
from ._env import DEVICE, read, read_table, write_correction, model_path
from .pipeline import RegenerationPipeline

__all__ = ["RegenerationPipeline", "DEVICE", "read", "read_table", "write_correction", "model_path"]
__version__ = "v20-2026-10-20"
