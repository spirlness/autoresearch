"""Compatibility shim for the canonical entrypoint in entrypoints/prepare.py."""

from entrypoints import prepare as _impl
from entrypoints.prepare import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(_impl.main())
