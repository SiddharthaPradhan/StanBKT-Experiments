#!/usr/bin/env python3
"""Quick pyBKT backend sanity check.

Prints whether pyBKT is using compiled extensions (fast) or Python fallbacks.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import sys


REQUIRED_FAST_MODULES: dict[str, str] = {
    "E_step": "pyBKT.fit.E_step",
    "predict_onestep_states": "pyBKT.fit.predict_onestep_states",
    "synthetic_data_helper": "pyBKT.generate.synthetic_data_helper",
}


def _is_compiled_path(path: str | None) -> bool:
    if not path:
        return False
    return path.endswith((".so", ".pyd", ".dylib"))


def _check_extension_module(module_name: str) -> tuple[bool, str]:
    """Return (is_fast_cpp_module, detail_message)."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"import_error={type(exc).__name__}: {exc}"

    spec = getattr(mod, "__spec__", None)
    loader = getattr(spec, "loader", None)
    origin = getattr(spec, "origin", None)

    has_extension_loader = isinstance(loader, importlib.machinery.ExtensionFileLoader)
    has_compiled_origin = _is_compiled_path(origin)
    is_fast = has_extension_loader and has_compiled_origin

    detail = (
        f"origin={origin}; "
        f"loader={type(loader).__name__ if loader is not None else 'None'}; "
        f"compiled_origin={has_compiled_origin}; "
        f"extension_loader={has_extension_loader}"
    )
    return is_fast, detail


def get_fast_math_checks() -> dict[str, tuple[bool, str]]:
    """Return backend checks for required compiled pyBKT modules."""
    return {
        short: _check_extension_module(full)
        for short, full in REQUIRED_FAST_MODULES.items()
    }


def has_fast_math() -> bool:
    """True if all required compiled pyBKT modules are active."""
    checks = get_fast_math_checks()
    return all(ok for ok, _ in checks.values())


def require_fast_math() -> None:
    """Raise RuntimeError if compiled pyBKT fast math modules are unavailable."""
    checks = get_fast_math_checks()
    missing = [f"{name}: {detail}" for name, (ok, detail) in checks.items() if not ok]
    if not missing:
        return

    msg = "fast_cpp_backend_unavailable; " + " | ".join(missing)
    raise RuntimeError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pyBKT backend type")
    parser.add_argument(
        "--require-fast",
        action="store_true",
        help="Exit with code 1 if compiled extensions are not active.",
    )
    args = parser.parse_args()

    checks = get_fast_math_checks()

    fast_cpp = all(ok for ok, _ in checks.values())
    backend = "FAST_CPP" if fast_cpp else "PURE_PY"

    print(f"backend={backend}")
    for name, (ok, detail) in checks.items():
        print(f"{name}={'FAST' if ok else 'SLOW'}; {detail}")

    if args.require_fast and not fast_cpp:
        try:
            require_fast_math()
        except RuntimeError as exc:
            print(f"error={exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
