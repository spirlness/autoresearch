import re
import os
from typing import Dict, Any

from .analyzer import find_best_result
from .config import MUON_WARMUP_STEPS, WARMUP_RATIO


RESEARCH_EMBEDDING_LR_CANDIDATES = (0.4, 0.36, 0.44, 0.32, 0.48, 0.28, 0.52)
RESEARCH_WARMUP_RATIO_CANDIDATES = (0.03, 0.05, 0.07, 0.1)
RESEARCH_MUON_WARMUP_CANDIDATES = (50, 100, 150, 200)
RESEARCH_MAX_SEQ_LEN_CANDIDATES = (2048, 3072, 4096)
RESEARCH_WINDOW_PATTERNS = ("LLLL", "SSSL")


def _format_env_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _normalize_env_signature(env_vars: Dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in env_vars.items()))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ordered_candidates(current_value: float | int | None, candidates: tuple[Any, ...]) -> list[Any]:
    if current_value is None:
        return list(candidates)
    unique_candidates = list(dict.fromkeys(candidates))
    return [
        value
        for _, value in sorted(
            enumerate(unique_candidates),
            key=lambda item: (round(abs(item[1] - current_value), 8), item[0]),
        )
    ]


def _build_candidate_env(
    best_env_vars: Dict[str, str],
    warmup_defaults: Dict[str, str],
    **overrides: str,
) -> Dict[str, str]:
    candidate_env_vars = dict(best_env_vars)
    candidate_env_vars.update(warmup_defaults)
    candidate_env_vars.update({key: str(value) for key, value in overrides.items()})
    return candidate_env_vars


def suggest_research_env_vars(results: list[dict[str, Any]]) -> Dict[str, str]:
    """Suggest the next env overrides by exploring around the best successful run so far."""
    tried_signatures = {
        _normalize_env_signature(result.get("applied_env_vars", {})) for result in results
    }

    best_result = find_best_result(results)
    best_env_vars = (
        {key: str(value) for key, value in best_result.get("applied_env_vars", {}).items()}
        if best_result is not None
        else {}
    )
    warmup_defaults = {
        "WARMUP_RATIO": _format_env_value(WARMUP_RATIO),
        "MUON_WARMUP_STEPS": str(MUON_WARMUP_STEPS),
    }
    best_config = (
        dict(best_result.get("summary", {}).get("config", {})) if best_result is not None else {}
    )

    current_embedding_lr = _coerce_float(
        best_env_vars.get("EMBEDDING_LR"), RESEARCH_EMBEDDING_LR_CANDIDATES[0]
    )
    current_warmup_ratio = _coerce_float(best_env_vars.get("WARMUP_RATIO"), WARMUP_RATIO)
    current_muon_warmup = _coerce_int(
        best_env_vars.get("MUON_WARMUP_STEPS"), MUON_WARMUP_STEPS
    )
    current_device_batch_size = _coerce_int(
        best_env_vars.get("DEVICE_BATCH_SIZE"),
        _coerce_int(best_config.get("device_batch_size"), 0),
    )
    current_max_seq_len = _coerce_int(
        best_env_vars.get("MAX_SEQ_LEN"),
        _coerce_int(best_config.get("max_seq_len"), RESEARCH_MAX_SEQ_LEN_CANDIDATES[0]),
    )
    current_window_pattern = str(
        best_env_vars.get("WINDOW_PATTERN", best_config.get("window_pattern", "LLLL"))
    ).upper()

    for embedding_lr in _ordered_candidates(
        current_embedding_lr, RESEARCH_EMBEDDING_LR_CANDIDATES
    ):
        candidate_env_vars = _build_candidate_env(
            best_env_vars,
            warmup_defaults,
            EMBEDDING_LR=_format_env_value(embedding_lr),
        )
        if _normalize_env_signature(candidate_env_vars) not in tried_signatures:
            return candidate_env_vars

    for warmup_ratio in _ordered_candidates(
        current_warmup_ratio, RESEARCH_WARMUP_RATIO_CANDIDATES
    ):
        candidate_env_vars = _build_candidate_env(
            best_env_vars,
            warmup_defaults,
            WARMUP_RATIO=_format_env_value(warmup_ratio),
        )
        if _normalize_env_signature(candidate_env_vars) not in tried_signatures:
            return candidate_env_vars

    for muon_warmup_steps in _ordered_candidates(
        current_muon_warmup, RESEARCH_MUON_WARMUP_CANDIDATES
    ):
        candidate_env_vars = _build_candidate_env(
            best_env_vars,
            warmup_defaults,
            MUON_WARMUP_STEPS=str(int(muon_warmup_steps)),
        )
        if _normalize_env_signature(candidate_env_vars) not in tried_signatures:
            return candidate_env_vars

    if current_device_batch_size > 0:
        for batch_size in (current_device_batch_size + 1, max(1, current_device_batch_size - 1)):
            candidate_env_vars = _build_candidate_env(
                best_env_vars,
                warmup_defaults,
                DEVICE_BATCH_SIZE=str(batch_size),
            )
            if _normalize_env_signature(candidate_env_vars) not in tried_signatures:
                return candidate_env_vars

    for max_seq_len in _ordered_candidates(current_max_seq_len, RESEARCH_MAX_SEQ_LEN_CANDIDATES):
        candidate_env_vars = _build_candidate_env(
            best_env_vars,
            warmup_defaults,
            MAX_SEQ_LEN=str(int(max_seq_len)),
        )
        if _normalize_env_signature(candidate_env_vars) not in tried_signatures:
            return candidate_env_vars

    for window_pattern in [current_window_pattern, *RESEARCH_WINDOW_PATTERNS]:
        candidate_env_vars = _build_candidate_env(
            best_env_vars,
            warmup_defaults,
            WINDOW_PATTERN=window_pattern,
        )
        if _normalize_env_signature(candidate_env_vars) not in tried_signatures:
            return candidate_env_vars

    fallback_env_vars = dict(best_env_vars)
    fallback_env_vars.update(warmup_defaults)
    return fallback_env_vars


def mutate_config(file_path: str, mutations: Dict[str, Any]) -> bool:
    """Mutate global constants in a python file using regex."""
    if not os.path.exists(file_path):
        return False
        
    with open(file_path, "r") as f:
        content = f.read()
        
    new_content = content
    for key, value in mutations.items():
        # Match "KEY = value" or "KEY: type = value"
        # Handles numbers, strings, tuples
        pattern = rf"^({key}\s*(?::\s*[\w\[\], ]+)?\s*=\s*).*?$"
        
        # Replacement value formatting
        if isinstance(value, str):
            val_str = f'"{value}"'
        else:
            val_str = str(value)
            
        new_content = re.sub(pattern, rf"\g<1>{val_str}", new_content, flags=re.MULTILINE)
        
    if new_content != content:
        with open(file_path, "w") as f:
            f.write(new_content)
        return True
        
    return False
