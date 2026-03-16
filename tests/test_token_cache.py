import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from autoresearch_trainer.token_cache import (
    _cache_dtype,
    _meta_matches,
)


def test_cache_dtype():
    assert _cache_dtype(100) == np.dtype(np.uint16)
    assert _cache_dtype(65535) == np.dtype(np.uint16)
    assert _cache_dtype(65536) == np.dtype(np.uint32)
    assert _cache_dtype(4294967295) == np.dtype(np.uint32)
    assert _cache_dtype(4294967296) == np.dtype(np.uint64)


def test_meta_matches():
    meta = {
        "version": 1,
        "fingerprint": "abc",
        "dtype_name": "uint16",
        "num_tokens": 100
    }

    cache_path = MagicMock(spec=Path)
    cache_path.exists.return_value = True

    stat_mock = MagicMock()
    stat_mock.st_size = 100 * 2  # 100 tokens * 2 bytes (uint16)
    cache_path.stat.return_value = stat_mock

    assert _meta_matches(meta, cache_path, fingerprint="abc", dtype_name="uint16")


def test_meta_matches_wrong_version():
    meta = {
        "version": 2,
        "fingerprint": "abc",
        "dtype_name": "uint16",
        "num_tokens": 100
    }
    cache_path = MagicMock(spec=Path)
    assert not _meta_matches(meta, cache_path, fingerprint="abc", dtype_name="uint16")


def test_meta_matches_wrong_fingerprint():
    meta = {
        "version": 1,
        "fingerprint": "def",
        "dtype_name": "uint16",
        "num_tokens": 100
    }
    cache_path = MagicMock(spec=Path)
    assert not _meta_matches(meta, cache_path, fingerprint="abc", dtype_name="uint16")


def test_meta_matches_missing_file():
    meta = {
        "version": 1,
        "fingerprint": "abc",
        "dtype_name": "uint16",
        "num_tokens": 100
    }
    cache_path = MagicMock(spec=Path)
    cache_path.exists.return_value = False
    assert not _meta_matches(meta, cache_path, fingerprint="abc", dtype_name="uint16")


def test_meta_matches_wrong_size():
    meta = {
        "version": 1,
        "fingerprint": "abc",
        "dtype_name": "uint16",
        "num_tokens": 100
    }
    cache_path = MagicMock(spec=Path)
    cache_path.exists.return_value = True

    stat_mock = MagicMock()
    stat_mock.st_size = 50 * 2  # Wrong size
    cache_path.stat.return_value = stat_mock

    assert not _meta_matches(meta, cache_path, fingerprint="abc", dtype_name="uint16")
