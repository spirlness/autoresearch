import re
import os
from typing import Dict, Any

from .analyzer import find_best_result
from .config import MUON_WARMUP_STEPS, WARMUP_RATIO


RESEARCH_EMBEDDING_LR_CANDIDATES = (0.4, 0.36, 0.44, 0.32, 0.48, 0.28, 0.52)


def _format_env_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _normalize_env_signature(env_vars: Dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in env_vars.items()))


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

    for embedding_lr in RESEARCH_EMBEDDING_LR_CANDIDATES:
        candidate_env_vars = dict(best_env_vars)
        candidate_env_vars.update(warmup_defaults)
        candidate_env_vars["EMBEDDING_LR"] = _format_env_value(embedding_lr)
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
