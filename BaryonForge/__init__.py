from . import Profiles, Runners, utils

from .Profiles import *
from .Runners  import *
from .utils    import *


try:
    from ._version import version as __version__
except Exception:
    __version__ = "0.0.0"

# Optional: expose a short commit SHA if present in PEP 440 local part (+g<sha>)
def _extract_commit(v: str):
    if "+g" in v:
        after = v.split("+g", 1)[1]
        return after.split(".", 1)[0]
    return None

__commit__ = _extract_commit(__version__)