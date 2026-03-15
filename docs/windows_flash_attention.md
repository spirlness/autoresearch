# Windows Flash Attention and Compile Notes

This project keeps a single source of truth for the Windows-specific setup needed to run the validated training profiles.

## What is pinned

- Python `3.12`
- PyTorch `2.10.0`
- CUDA `13.0`
- `triton-windows==3.6.0.post26`
- Flash Attention wheel in `vendor/flash_attn-2.8.3+cu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl`

The wheel is intentionally kept inside `vendor/` because the Windows path is part of the reproducible environment for `uv sync`.

## Install / verify

```bash
uv sync
uv run python verify_flash_attn.py
```

Expected outcome:

- `flash_attn` imports successfully
- CUDA device is visible
- the smoke test forward pass succeeds

## Training entrypoints

```bash
uv run train.py
uv run python -m autoresearch_trainer
uv run train.py --benchmark-steps 20
uv run train.py --experiment-profile mfu50 --benchmark-steps 20
```

## Notes

- `train.py` is a thin entrypoint; the real training runtime lives in `autoresearch_trainer/`.
- The current throughput-first and MFU-first defaults are documented in `benchmarks/KEY_FINDINGS.md`.
- The detailed benchmark table is in `benchmarks/SUMMARY_2026-03-15.md`.
