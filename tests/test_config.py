import os
import pytest
from unittest.mock import patch
from autoresearch_trainer.config import env_override_int, env_override_str, env_override_float

def test_env_override_int_default():
    with patch.dict(os.environ, {}, clear=True):
        assert env_override_int("TEST_INT", 42) == 42

def test_env_override_int_override():
    with patch.dict(os.environ, {"TEST_INT": "100"}):
        assert env_override_int("TEST_INT", 42) == 100

def test_env_override_int_invalid():
    with patch.dict(os.environ, {"TEST_INT": "invalid"}):
        with pytest.raises(ValueError):
            env_override_int("TEST_INT", 42)

def test_env_override_int_empty_string():
    with patch.dict(os.environ, {"TEST_INT": ""}):
        with pytest.raises(ValueError):
            env_override_int("TEST_INT", 42)

def test_env_override_str_default():
    with patch.dict(os.environ, {}, clear=True):
        assert env_override_str("TEST_STR", "default") == "default"

def test_env_override_str_override():
    with patch.dict(os.environ, {"TEST_STR": "override"}):
        assert env_override_str("TEST_STR", "default") == "override"

def test_env_override_str_empty_string():
    with patch.dict(os.environ, {"TEST_STR": ""}):
        assert env_override_str("TEST_STR", "default") == ""

def test_env_override_float_default():
    with patch.dict(os.environ, {}, clear=True):
        assert env_override_float("TEST_FLOAT", 3.14) == 3.14

def test_env_override_float_override():
    with patch.dict(os.environ, {"TEST_FLOAT": "2.71"}):
        assert env_override_float("TEST_FLOAT", 3.14) == 2.71

def test_env_override_float_invalid():
    with patch.dict(os.environ, {"TEST_FLOAT": "invalid"}):
        with pytest.raises(ValueError):
            env_override_float("TEST_FLOAT", 3.14)

def test_env_override_float_empty_string():
    with patch.dict(os.environ, {"TEST_FLOAT": ""}):
        with pytest.raises(ValueError):
            env_override_float("TEST_FLOAT", 3.14)
