import os

import pytest
from unittest.mock import MagicMock, patch

from autoresearch_trainer.utils.platform import (
    load_windows_msvc_env,
    maybe_patch_msvc_utf8_help,
)
from torch._inductor import cpp_builder


def test_load_windows_msvc_env_uses_cmd_and_sets_environment():
    with patch("os.name", "nt"), patch.dict(os.environ, {}, clear=True), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value = MagicMock(
            stdout="INCLUDE=C:\\vc\\include\nPATH=C:\\vc\\bin\n"
        )

        load_windows_msvc_env(r"C:\VS\VC\Auxiliary\Build\vcvars64.bat")

        args, kwargs = mock_run.call_args
        assert args[0][:4] == ["cmd.exe", "/d", "/s", "/c"]
        assert "&& set" in args[0][4]
        assert os.environ["INCLUDE"] == r"C:\vc\include"
        assert os.environ["PATH"] == r"C:\vc\bin"


def test_maybe_patch_msvc_utf8_help_runtime_error():
    with patch("os.name", "nt"), patch(
        "subprocess.check_output"
    ) as mock_subprocess, patch.object(
        cpp_builder, "SUBPROCESS_DECODE_ARGS", ("mbcs",)
    ):
        mock_help_output = MagicMock()
        mock_subprocess.return_value = mock_help_output

        # The decode method raises UnicodeDecodeError both times
        def decode_side_effect(*args, **kwargs):
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "mock error")

        mock_help_output.decode.side_effect = decode_side_effect

        with pytest.raises(RuntimeError) as exc_info:
            maybe_patch_msvc_utf8_help("mock_compiler.exe")

        # Verify the error message
        assert "mock_compiler.exe" in str(exc_info.value)
        assert "could not be decoded using" in str(exc_info.value)


def test_maybe_patch_msvc_utf8_help_success():
    with patch("os.name", "nt"), patch(
        "subprocess.check_output"
    ) as mock_subprocess, patch.object(
        cpp_builder, "SUBPROCESS_DECODE_ARGS", ("mbcs",)
    ), patch.object(
        cpp_builder._is_msvc_cl, "cache_clear"
    ) as mock_cache_clear:
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

        result = maybe_patch_msvc_utf8_help("mock_compiler.exe")

        # Verify the result and side effects
        assert result == "utf-8"
        assert cpp_builder.SUBPROCESS_DECODE_ARGS == ("utf-8", "ignore")
        mock_cache_clear.assert_called_once()
