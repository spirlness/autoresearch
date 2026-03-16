import pytest
from scripts.check_push_artifacts import format_bytes

@pytest.mark.parametrize(
    "num_bytes, expected",
    [
        (0, "0.0 B"),
        (1, "1.0 B"),
        (100, "100.0 B"),
        (1023, "1023.0 B"),
        (1024, "1.0 KB"),
        (1048575, "1024.0 KB"), # Just under 1 MB
        (1048576, "1.0 MB"),
        (1073741823, "1024.0 MB"), # Just under 1 GB
        (1073741824, "1.0 GB"),
        (1536, "1.5 KB"), # 1.5 KB
        (1572864, "1.5 MB"), # 1.5 MB
        (1610612736, "1.5 GB"), # 1.5 GB
        (1024 * 1024 * 1024 * 1024, "1024.0 GB"), # 1 TB should map to GB since it's the largest unit
    ]
)
def test_format_bytes(num_bytes: int, expected: str):
    """Test format_bytes for various sizes spanning multiple orders of magnitude."""
    assert format_bytes(num_bytes) == expected
