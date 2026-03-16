import pytest
from unittest.mock import patch
from autoresearch_trainer.compile import validate_compile_backend, validate_compile_mode

def test_validate_compile_backend_off():
    # Should pass silently
    validate_compile_backend("off")

@patch('autoresearch_trainer.compile.AVAILABLE_COMPILE_BACKENDS', {'inductor', 'eager'})
@patch('autoresearch_trainer.compile.EXTRA_COMPILE_BACKENDS', {'aot_eager'})
def test_validate_compile_backend_available():
    # Should pass silently
    validate_compile_backend("eager")

@patch('autoresearch_trainer.compile.AVAILABLE_COMPILE_BACKENDS', {'inductor', 'eager'})
@patch('autoresearch_trainer.compile.EXTRA_COMPILE_BACKENDS', {'aot_eager'})
def test_validate_compile_backend_extra():
    # Should pass silently
    validate_compile_backend("aot_eager")

@patch('autoresearch_trainer.compile.AVAILABLE_COMPILE_BACKENDS', {'inductor', 'eager'})
@patch('autoresearch_trainer.compile.EXTRA_COMPILE_BACKENDS', {'aot_eager'})
def test_validate_compile_backend_invalid():
    with pytest.raises(RuntimeError, match="Compile backend 'invalid' is unavailable. Available backends: aot_eager, eager, inductor"):
        validate_compile_backend("invalid")

@patch('autoresearch_trainer.compile.AVAILABLE_COMPILE_BACKENDS', {'inductor', 'eager'})
@patch('autoresearch_trainer.compile.EXTRA_COMPILE_BACKENDS', {'aot_eager'})
@patch('autoresearch_trainer.compile.HAS_TRITON', False)
def test_validate_compile_backend_inductor_no_triton():
    with pytest.raises(RuntimeError, match="Compile backend 'inductor' requires Triton in this environment"):
        validate_compile_backend("inductor")

@patch('autoresearch_trainer.compile.AVAILABLE_COMPILE_BACKENDS', {'inductor', 'eager'})
@patch('autoresearch_trainer.compile.EXTRA_COMPILE_BACKENDS', {'aot_eager'})
@patch('autoresearch_trainer.compile.HAS_TRITON', True)
def test_validate_compile_backend_inductor_with_triton():
    # Should pass silently
    validate_compile_backend("inductor")

def test_validate_compile_mode_not_inductor():
    # Should pass silently
    validate_compile_mode("eager", "any_mode")

@patch('autoresearch_trainer.compile.inductor', None)
def test_validate_compile_mode_no_inductor_module():
    # Should pass silently
    validate_compile_mode("inductor", "any_mode")

class MockInductor:
    pass

@patch('autoresearch_trainer.compile.inductor', MockInductor())
@patch('autoresearch_trainer.compile.AVAILABLE_INDUCTOR_MODES', ['default', 'max-autotune'])
def test_validate_compile_mode_valid():
    # Should pass silently
    validate_compile_mode("inductor", "max-autotune")

@patch('autoresearch_trainer.compile.inductor', MockInductor())
@patch('autoresearch_trainer.compile.AVAILABLE_INDUCTOR_MODES', ['default', 'max-autotune'])
def test_validate_compile_mode_invalid():
    with pytest.raises(RuntimeError, match="Compile mode 'invalid_mode' is unavailable. Available modes: default, max-autotune"):
        validate_compile_mode("inductor", "invalid_mode")
