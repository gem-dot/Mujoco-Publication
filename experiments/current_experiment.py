"""Compatibility notice for the retired Phase-controller experiment.

The maintained one-click experiment now lives in ``controllers/YAC.py``.
This module intentionally contains no runnable controller configuration so a
legacy Phase experiment cannot be launched accidentally after its controller
was removed.
"""

from controllers.YAC import (
    DEFAULT_DURATION_S,
    DEFAULT_MODE,
    DEFAULT_PATHOLOGY,
    DEFAULT_VIEWER,
)


MIGRATED_ENTRYPOINT = "controllers/YAC.py"
