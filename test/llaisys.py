from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_pkg_dir = Path(__file__).resolve().parents[1] / "python" / "llaisys"
_spec = spec_from_file_location(
    __name__,
    _pkg_dir / "__init__.py",
    submodule_search_locations=[str(_pkg_dir)],
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load local llaisys package from {_pkg_dir}")

_module = module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
