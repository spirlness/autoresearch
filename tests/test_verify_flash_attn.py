import pytest
from unittest.mock import MagicMock, patch

import verify_flash_attn

@pytest.fixture
def mock_success_env():
    """Mock a successful environment with flash_attn and torch.cuda available."""
    with patch("verify_flash_attn.FLASH_ATTN_IMPORT_ERROR", None), \
         patch("verify_flash_attn.flash_attn") as mock_flash_attn, \
         patch("verify_flash_attn.flash_attn_func") as mock_flash_attn_func, \
         patch("verify_flash_attn.torch.cuda.is_available", return_value=True), \
         patch("verify_flash_attn.torch.version.cuda", "12.1"), \
         patch("verify_flash_attn.torch.cuda.get_device_name", return_value="Mock GPU"), \
         patch("verify_flash_attn.torch.cuda.get_device_properties") as mock_props, \
         patch("verify_flash_attn.torch.cuda.synchronize"), \
         patch("verify_flash_attn.torch.randn") as mock_randn:

        mock_flash_attn.__version__ = "2.8.3"

        mock_props_obj = MagicMock()
        mock_props_obj.total_memory = 24 * 1024**3
        mock_props.return_value = mock_props_obj

        # Mock torch.randn to just return a dummy tensor on CPU without actually trying to use CUDA
        # (which throws "Found no NVIDIA driver on your system").
        mock_randn.return_value = MagicMock()

        # Mock the forward pass shape
        mock_output = MagicMock()
        mock_output.shape = (2, 128, 8, 64)
        mock_flash_attn_func.return_value = mock_output

        yield {
            "flash_attn": mock_flash_attn,
            "flash_attn_func": mock_flash_attn_func,
        }

def test_verify_installation_import_error():
    """Test verification fails when flash_attn cannot be imported."""
    with patch("verify_flash_attn.FLASH_ATTN_IMPORT_ERROR", ImportError("No module named flash_attn")):
        assert verify_flash_attn.verify_installation() is False

def test_verify_installation_success(mock_success_env):
    """Test verification succeeds when everything is correctly installed and functioning."""
    assert verify_flash_attn.verify_installation() is True

def test_verify_installation_functional_test_failure(mock_success_env):
    """Test verification fails if the functional test throws an exception."""
    mock_success_env["flash_attn_func"].side_effect = RuntimeError("CUDA Out of Memory")

    assert verify_flash_attn.verify_installation() is False

def test_main_success(mock_success_env):
    """Test main returns 0 on success."""
    assert verify_flash_attn.main() == 0

def test_main_failure():
    """Test main returns 1 on failure."""
    with patch("verify_flash_attn.verify_installation", return_value=False):
        assert verify_flash_attn.main() == 1

def test_verify_installation_no_cuda(mock_success_env):
    """Test verification handles environment where CUDA isn't strictly 'available'."""
    with patch("verify_flash_attn.torch.cuda.is_available", return_value=False):
        # Even if torch.randn is called with device="cuda", we mock it to raise RuntimeError
        with patch("verify_flash_attn.torch.randn", side_effect=RuntimeError("Expected a 'cuda' device type")):
            assert verify_flash_attn.verify_installation() is False
