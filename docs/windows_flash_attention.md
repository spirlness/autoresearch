# Windows Flash Attention and Compile Notes

This project keeps a single source of truth for the Windows-specific setup needed to run the validated training profiles.

## What is pinned

- Python `3.12`
- PyTorch `2.10.0`
- CUDA `13.0`
- `triton-windows==3.6.0.post26`
- Flash Attention wheel `flash_attn-2.8.3+cu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl`

The wheel is fetched directly by `uv` from its pinned download URL and verified against SHA256 `d1e76adda81f57d9a300b9e8d71b9c467e07923e0d8e5baa5d90e2007815ec93`. This keeps the Git repository free of large binary history while preserving a reproducible install on Windows.

## Install / verify

```bash
uv sync
uv run python verify_flash_attn.py
```

If you want an extra local backup copy, you can keep the wheel under `vendor/`, but it is no longer required for `uv sync` and should remain untracked.

Expected outcome:

- `flash_attn` imports successfully
- CUDA device is visible
- the smoke test forward pass succeeds

## Training entrypoints

```bash
uv run train.py
uv run python -m autoresearch_trainer
uv run train.py --benchmark-steps 20
uv run train.py --experiment-profile throughput --benchmark-steps 20
uv run train.py --experiment-profile mfu50 --benchmark-steps 20
```

## Notes

- `train.py` is a thin entrypoint; the real training runtime lives in `autoresearch_trainer/`.
- The current default baseline is the deeper near-50%-MFU profile with MLP-only checkpointing; the old throughput-oriented profile is preserved as `throughput`.
- The detailed benchmark table is in `benchmarks/SUMMARY_2026-03-15.md`.
