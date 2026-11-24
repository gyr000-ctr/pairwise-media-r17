"""camcurve: Catmull-Rom camera paths with stable orthonormal orientation."""

from .core import *  # noqa: F401,F403
from .core import CamCurve, __all__  # noqa: F401  (re-export the public list)


def __getattr__(name):
    # Delegate unknown attributes (notably speculative *Error names) to core's
    # PEP 562 resolver so ``from camcurve import SomeError`` keeps working.
    from . import core

    return getattr(core, name)
