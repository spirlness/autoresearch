import pytest
from unittest.mock import MagicMock, patch
from autoresearch_trainer.runner import run_research_loop

def test_run_research_loop_3_iterations():
    with patch("autoresearch_trainer.runner.run_experiment") as mock_run, \
         patch("autoresearch_trainer.runner.get_summary") as mock_summary, \
         patch("autoresearch_trainer.runner.mutate_config") as mock_mutate:
        
        # Iteration 1: Success
        # Iteration 2: Success
        # Iteration 3: Success
        mock_run.return_value = {"status": "success", "elapsed": 300}
        mock_summary.return_value = {"val_bpb": 1.2, "loss": 9.0}
        mock_mutate.return_value = True
        
        results = run_research_loop(iterations=3)
        
        assert len(results) == 3
        assert mock_run.call_count == 3
        assert mock_summary.call_count == 3
        assert mock_mutate.call_count == 2 # Only 2 mutations between 3 runs
