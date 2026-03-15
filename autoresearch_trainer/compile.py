import importlib
from dataclasses import dataclass
from typing import Any

import torch
# In Torch 2.10+, inductor has moved:
try:
    from torch import inductor
except ImportError:
    try:
        from torch._inductor import inductor
    except ImportError:
        inductor = None

from .utils.platform import setup_platform_environment


HAS_TRITON = importlib.util.find_spec("triton") is not None
AVAILABLE_COMPILE_BACKENDS = set(torch.compiler.list_backends())
EXTRA_COMPILE_BACKENDS = {"aot_eager"}
AVAILABLE_INDUCTOR_MODES = sorted(inductor.list_mode_options().keys()) if inductor else []


@dataclass(frozen=True)
class CompilePreparation:
    msvc_cl_path: str | None
    msvc_help_encoding: str | None


def resolve_compile_backend(requested_backend: str) -> str:
    if requested_backend == "auto":
        return "inductor" if HAS_TRITON else "off"
    return requested_backend


def resolve_optimizer_compile_backend(requested_backend: str, model_backend: str) -> str:
    if requested_backend == "auto":
        return "inductor" if model_backend == "inductor" else "off"
    return requested_backend


def validate_compile_backend(compile_backend: str) -> None:
    if compile_backend == "off":
        return
    if compile_backend not in AVAILABLE_COMPILE_BACKENDS and compile_backend not in EXTRA_COMPILE_BACKENDS:
        available = ", ".join(sorted(AVAILABLE_COMPILE_BACKENDS | EXTRA_COMPILE_BACKENDS))
        raise RuntimeError(f"Compile backend '{compile_backend}' is unavailable. Available backends: {available}")
    if compile_backend == "inductor" and not HAS_TRITON:
        raise RuntimeError("Compile backend 'inductor' requires Triton in this environment")


def validate_compile_mode(compile_backend: str, compile_mode: str) -> None:
    if compile_backend != "inductor":
        return
    if not inductor:
        return # Skip if inductor module discovery failed
    if compile_mode not in AVAILABLE_INDUCTOR_MODES:
        available = ", ".join(AVAILABLE_INDUCTOR_MODES)
        raise RuntimeError(f"Compile mode '{compile_mode}' is unavailable. Available modes: {available}")


def prepare_compile_environment(
    *,
    model_backend: str,
    optimizer_backend: str,
) -> CompilePreparation:
    """Prepare the environment for compilation, handling platform-specific requirements."""
    msvc_cl_path = None
    msvc_help_encoding = None
    
    if model_backend == "inductor" or optimizer_backend == "inductor":
        platform_info = setup_platform_environment()
        msvc_cl_path = platform_info.msvc_cl_path
        msvc_help_encoding = platform_info.msvc_help_encoding
        
    return CompilePreparation(
        msvc_cl_path=msvc_cl_path,
        msvc_help_encoding=msvc_help_encoding,
    )


def maybe_compile_function(fn, *, backend: str, compile_mode: str, **kwargs):
    if backend == "off":
        return fn
    # Keep the compile call in one helper so the model/runtime code can switch
    # between eager and compiled paths without repeating backend plumbing.
    compile_kwargs: dict[str, Any] = {"backend": backend, **kwargs}
    if backend == "inductor":
        compile_kwargs["mode"] = compile_mode
    return torch.compile(fn, **compile_kwargs)
