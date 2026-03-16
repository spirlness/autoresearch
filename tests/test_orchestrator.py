import pytest
import subprocess
from unittest.mock import MagicMock, patch
from autoresearch_trainer.orchestrator import run_experiment

def test_run_experiment_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        result = run_experiment(timeout=300)
        
        assert result["status"] == "success"
        assert result["returncode"] == 0
        mock_run.assert_called_once()
        # Check if timeout was passed to subprocess.run
        args, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 300

def test_run_experiment_timeout():
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["uv", "run", "train.py"], timeout=300)
        
        result = run_experiment(timeout=300)
        
        assert result["status"] == "timeout"
        mock_run.assert_called_once()

def test_run_experiment_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        
        result = run_experiment(timeout=300)
        
        assert result["status"] == "failed"
        assert result["returncode"] == 1
