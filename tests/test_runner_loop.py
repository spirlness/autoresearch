import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_trainer.runner import (
    build_research_loop_extra_args,
    compute_oom_recovery_settings,
    persist_research_artifacts,
    render_next_research_run_markdown,
    run_research_loop,
)


def test_run_research_loop_3_iterations():
    with patch("autoresearch_trainer.runner.run_experiment") as mock_run, \
         patch("autoresearch_trainer.runner.get_summary") as mock_summary, \
         patch("autoresearch_trainer.runner.persist_research_artifacts") as mock_persist:

        mock_run.return_value = {"status": "success", "elapsed": 300}
        mock_summary.side_effect = [
            {"val_bpb": 1.2, "loss": 9.0},
            {"val_bpb": 1.1, "loss": 8.5},
            {"val_bpb": 1.3, "loss": 8.8},
        ]
        mock_persist.return_value = {"history_path": "results/research_loop/history.json"}

        results = run_research_loop(
            iterations=3,
            timeout=None,
            profile="throughput",
            extra_args=["--seed", "123"],
        )

        assert len(results) == 3
        assert mock_run.call_count == 3
        assert mock_summary.call_count == 3
        assert mock_persist.call_count == 3
        
        # Check that environment overrides are being passed along correctly
        # First call has no env vars set by the loop
        assert results[0]["applied_env_vars"] == {}
        # Subsequent calls explore warmup-aware candidates around the best run so far
        assert "EMBEDDING_LR" in results[1]["applied_env_vars"]
        assert results[1]["applied_env_vars"]["EMBEDDING_LR"] == "0.4"
        assert results[1]["applied_env_vars"]["WARMUP_RATIO"] == "0.05"
        assert results[1]["applied_env_vars"]["MUON_WARMUP_STEPS"] == "100"
        assert results[2]["applied_env_vars"]["EMBEDDING_LR"] == "0.36"
        assert results[0]["progress"]["current_is_best"] is True
        assert results[1]["progress"]["current_is_best"] is True
        assert results[2]["progress"]["best_iteration"] == 2
        assert results[2]["recommended_next_env_vars"]["EMBEDDING_LR"] == "0.44"
        assert results[2]["artifact_paths"]["history_path"] == "results/research_loop/history.json"

        first_call = mock_run.call_args_list[0]
        assert first_call.kwargs["timeout"] is None
        assert first_call.kwargs["profile"] == "throughput"
        assert first_call.kwargs["extra_args"] == ["--seed", "123"]


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


def test_render_next_research_run_markdown_includes_best_and_next_env():
    markdown = render_next_research_run_markdown(
        {
            "latest_iteration": 3,
            "best_iteration": 2,
            "current_is_best": False,
            "best_summary": {"val_bpb": 1.1},
            "best_env_vars": {"EMBEDDING_LR": "0.4"},
        },
        {"EMBEDDING_LR": "0.44", "WARMUP_RATIO": "0.05"},
    )

    assert "Best iteration: 2" in markdown
    assert "`EMBEDDING_LR=0.44`" in markdown
    assert "`val_bpb`: `1.1`" in markdown


def test_persist_research_artifacts_writes_resume_files(tmp_path):
    results = [
        {
            "iteration": 1,
            "experiment": {"status": "success"},
            "summary": {"val_bpb": 1.2, "loss": 9.0},
            "applied_env_vars": {},
            "recommended_next_env_vars": {
                "EMBEDDING_LR": "0.4",
                "WARMUP_RATIO": "0.05",
            },
        }
    ]

    artifact_paths = persist_research_artifacts(results, state_dir=tmp_path)

    history_path = Path(artifact_paths["history_path"])
    best_run_path = Path(artifact_paths["best_run_path"])
    next_env_path = Path(artifact_paths["next_env_path"])
    next_run_markdown_path = Path(artifact_paths["next_run_markdown_path"])

    assert history_path.exists()
    assert best_run_path.exists()
    assert next_env_path.exists()
    assert next_run_markdown_path.exists()

    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    best_payload = json.loads(best_run_path.read_text(encoding="utf-8"))
    next_env_payload = json.loads(next_env_path.read_text(encoding="utf-8"))

    assert history_payload["iterations"][0]["iteration"] == 1
    assert best_payload["progress"]["best_iteration"] == 1
    assert next_env_payload["recommended_next_env_vars"]["EMBEDDING_LR"] == "0.4"
    assert "Recommended Next Env Vars" in next_run_markdown_path.read_text(encoding="utf-8")
