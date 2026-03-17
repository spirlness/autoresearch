"""Compatibility shim for the canonical entrypoint in entrypoints/verify_flash_attn.py."""

from entrypoints import verify_flash_attn as _impl
from entrypoints.verify_flash_attn import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(_impl.main())
