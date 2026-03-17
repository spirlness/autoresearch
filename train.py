"""Compatibility shim for the canonical entrypoint in entrypoints/train.py."""

from entrypoints.train import main


if __name__ == "__main__":
    raise SystemExit(main())
