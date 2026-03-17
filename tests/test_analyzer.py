import json
from unittest.mock import mock_open, patch
from autoresearch_trainer.analyzer import (
    build_research_progress_report,
    find_best_result,
    get_summary,
    parse_ledger,
    parse_metrics,
    score_summary,
)


def test_parse_metrics():
    data = [
        {"step": 0, "loss": 10.0, "end_to_end": {"tok_per_sec": 1000, "mfu": 10.0}},
        {"step": 1, "loss": 9.0, "end_to_end": {"tok_per_sec": 1100, "mfu": 11.0}},
    ]
    read_data = "".join(json.dumps(entry) + "\n" for entry in data)

    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data=read_data)
    ):
        parsed = parse_metrics("metrics.jsonl")

    assert len(parsed) == 2
    assert parsed[-1]["loss"] == 9.0
    assert parsed[-1]["end_to_end"]["tok_per_sec"] == 1100


def test_parse_ledger():
    data = [
        {"val_bpb": 1.234, "end_to_end_tok_per_sec": 1100, "peak_vram_mb": 5000},
    ]
    read_data = "".join(json.dumps(entry) + "\n" for entry in data)

    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data=read_data)
    ):
        parsed = parse_ledger("experiment_ledger.jsonl")

    assert len(parsed) == 1
    assert parsed[0]["val_bpb"] == 1.234


def test_get_summary():
    metrics_data = [{"step": 1, "loss": 9.0, "end_to_end": {"tok_per_sec": 1100}}]
    ledger_data = [
        {
            "val_bpb": 1.234,
            "end_to_end_tok_per_sec": 1100,
            "warmup_excluded_tok_per_sec": 1200,
            "warmup_excluded_mfu": 12.5,
            "config": {"depth": 9},
        }
    ]

    with patch(
        "autoresearch_trainer.analyzer.parse_metrics", return_value=metrics_data
    ), patch("autoresearch_trainer.analyzer.parse_ledger", return_value=ledger_data):
        summary = get_summary("metrics.jsonl", "experiment_ledger.jsonl")

    assert summary["val_bpb"] == 1.234
    assert summary["tok_per_sec"] == 1100
    assert summary["warmup_tok_per_sec"] == 1200
    assert summary["warmup_mfu"] == 12.5
    assert summary["config"] == {"depth": 9}
    assert "loss" in summary


def test_score_summary_prefers_lower_val_bpb_then_higher_throughput():
    better = {"val_bpb": 1.1, "tok_per_sec": 900}
    worse = {"val_bpb": 1.2, "tok_per_sec": 1100}

    assert score_summary(better) < score_summary(worse)


def test_find_best_result_ignores_failed_runs():
    results = [
        {"iteration": 1, "experiment": {"status": "failed"}, "summary": {"val_bpb": 0.9}},
        {"iteration": 2, "experiment": {"status": "success"}, "summary": {"val_bpb": 1.1}},
        {"iteration": 3, "experiment": {"status": "success"}, "summary": {"val_bpb": 1.2}},
    ]

    best = find_best_result(results)

    assert best is not None
    assert best["iteration"] == 2


def test_build_research_progress_report_marks_new_best():
    results = [
        {
            "iteration": 1,
            "experiment": {"status": "success"},
            "summary": {"val_bpb": 1.3},
            "applied_env_vars": {},
        },
        {
            "iteration": 2,
            "experiment": {"status": "success"},
            "summary": {"val_bpb": 1.1},
            "applied_env_vars": {"EMBEDDING_LR": "0.4"},
        },
    ]

    report = build_research_progress_report(results)

    assert report["latest_iteration"] == 2
    assert report["best_iteration"] == 2
    assert report["current_is_best"] is True
    assert report["best_env_vars"] == {"EMBEDDING_LR": "0.4"}
