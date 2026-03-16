import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock

from autoresearch_trainer.utils.platform import maybe_patch_msvc_utf8_help
from torch._inductor import cpp_builder

def test_maybe_patch_msvc_utf8_help_runtime_error(mocker):
    # Setup mocks
    mocker.patch("os.name", "nt")

    mock_subprocess = mocker.patch("subprocess.check_output")
    mock_help_output = MagicMock()
    mock_subprocess.return_value = mock_help_output

    # The decode method raises UnicodeDecodeError both times
    def decode_side_effect(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "mock error")

    mock_help_output.decode.side_effect = decode_side_effect

    # Mock cpp_builder.SUBPROCESS_DECODE_ARGS because it may be empty in some setups
    mocker.patch.object(cpp_builder, "SUBPROCESS_DECODE_ARGS", ("mbcs",))

    with pytest.raises(RuntimeError) as exc_info:
        maybe_patch_msvc_utf8_help("mock_compiler.exe")

    # Verify the error message
    assert "mock_compiler.exe" in str(exc_info.value)
    assert "could not be decoded using" in str(exc_info.value)


def test_maybe_patch_msvc_utf8_help_success(mocker):
    # Setup mocks
    mocker.patch("os.name", "nt")

    mock_subprocess = mocker.patch("subprocess.check_output")
    mock_help_output = MagicMock()
    mock_subprocess.return_value = mock_help_output

    # The first decode fails, the second one succeeds
    decode_calls = 0
    def decode_side_effect(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        if decode_calls == 1:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "mock error")
        return "success"

    mock_help_output.decode.side_effect = decode_side_effect

    # Mock cpp_builder.SUBPROCESS_DECODE_ARGS and _is_msvc_cl
    mocker.patch.object(cpp_builder, "SUBPROCESS_DECODE_ARGS", ("mbcs",))
    mocker.patch.object(cpp_builder._is_msvc_cl, "cache_clear")

    result = maybe_patch_msvc_utf8_help("mock_compiler.exe")

    # Verify the result and side effects
    assert result == "utf-8"
    assert cpp_builder.SUBPROCESS_DECODE_ARGS == ("utf-8", "ignore")
    cpp_builder._is_msvc_cl.cache_clear.assert_called_once()
