import pytest
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_trainer.runner import (
    build_research_loop_extra_args,
    compute_oom_recovery_settings,
    run_research_loop,
)

def test_run_research_loop_3_iterations():
    with patch("autoresearch_trainer.runner.run_experiment") as mock_run, \
         patch("autoresearch_trainer.runner.get_summary") as mock_summary:

        mock_run.return_value = {"status": "success", "elapsed": 300}
        mock_summary.return_value = {"val_bpb": 1.2, "loss": 9.0}

        results = run_research_loop(
            iterations=3,
            timeout=None,
            profile="throughput",
            extra_args=["--seed", "123"],
        )

        assert len(results) == 3
        assert mock_run.call_count == 3
        assert mock_summary.call_count == 3
        first_call = mock_run.call_args_list[0]
        second_call = mock_run.call_args_list[1]
        assert first_call.kwargs["timeout"] is None
        assert first_call.kwargs["profile"] == "throughput"
        assert first_call.kwargs["extra_args"] == ["--seed", "123"]
        assert first_call.kwargs["env_vars"] == {}
        assert second_call.kwargs["env_vars"] == {"EMBEDDING_LR": "0.4"}

def test_build_research_loop_extra_args():
    args = SimpleNamespace(
        benchmark_steps=20,
        compile_backend="inductor",
        compile_mode="default",
        compile_scope="trunk",
        optimizer_compile_backend="auto",
        grad_accum_steps=3,
        seed=123,
    )

    extra_args = build_research_loop_extra_args(args)

    assert extra_args == [
        "--benchmark-steps",
        "20",
        "--compile-backend",
        "inductor",
        "--compile-mode",
        "default",
        "--compile-scope",
        "trunk",
        "--optimizer-compile-backend",
        "auto",
        "--grad-accum-steps",
        "3",
        "--seed",
        "123",
    ]

def test_compute_oom_recovery_settings_uses_ceil_for_odd_batches():
    assert compute_oom_recovery_settings(5, 1) == (2, 3, 5, 6)
