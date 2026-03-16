from unittest.mock import patch
from autoresearch_trainer.runner import run_research_loop

def test_run_research_loop_3_iterations():
    with patch("autoresearch_trainer.runner.run_experiment") as mock_run, \
         patch("autoresearch_trainer.runner.get_summary") as mock_summary:
        
        # Iteration 1: Success
        # Iteration 2: Success
        # Iteration 3: Success
        mock_run.return_value = {"status": "success", "elapsed": 300}
        mock_summary.return_value = {"val_bpb": 1.2, "loss": 9.0}
        
        results = run_research_loop(iterations=3)
        
        assert len(results) == 3
        assert mock_run.call_count == 3
        assert mock_summary.call_count == 3

        # Check that environment overrides are being passed along correctly
        # First call has no env vars set by the loop
        assert results[0]["applied_env_vars"] == {}
        # Subsequent calls have the modified env vars
        assert "EMBEDDING_LR" in results[1]["applied_env_vars"]
        assert results[1]["applied_env_vars"]["EMBEDDING_LR"] == "0.4"
