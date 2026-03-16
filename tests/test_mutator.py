from unittest.mock import mock_open, patch
from autoresearch_trainer.mutator import mutate_config


def _run_mutation(content: str, mutations: dict[str, object]) -> tuple[bool, str]:
    file_mock = mock_open(read_data=content)
    written: list[str] = []

    def capture_write(data: str) -> int:
        written.append(data)
        return len(data)

    file_mock.return_value.write.side_effect = capture_write

    with patch("os.path.exists", return_value=True), patch("builtins.open", file_mock):
        changed = mutate_config("config.py", mutations)

    return changed, "".join(written)


def test_mutate_config():
    content = """
EMBEDDING_LR = 0.6
MATRIX_LR = 0.04
DEPTH = 8
"""

    changed, new_content = _run_mutation(
        content, {"EMBEDDING_LR": 0.5, "DEPTH": 12}
    )

    assert changed is True
    assert "EMBEDDING_LR = 0.5" in new_content
    assert "DEPTH = 12" in new_content
    assert "MATRIX_LR = 0.04" in new_content
    assert "EMBEDDING_LR = 0.6" not in new_content


def test_mutate_config_type_hint():
    changed, new_content = _run_mutation("LR: float = 0.1", {"LR": 0.2})

    assert changed is True
    assert "LR: float = 0.2" in new_content


def test_mutate_config_float_formatting():
    changed, new_content = _run_mutation("LR = 0.1", {"LR": 0.0001})

    assert changed is True
    assert "LR = 0.0001" in new_content
