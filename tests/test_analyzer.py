import json
from unittest.mock import mock_open, patch
from autoresearch_trainer.analyzer import parse_metrics, parse_ledger, get_summary

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
    ledger_data = [{"val_bpb": 1.234, "end_to_end_tok_per_sec": 1100}]

    with patch(
        "autoresearch_trainer.analyzer.parse_metrics", return_value=metrics_data
    ), patch("autoresearch_trainer.analyzer.parse_ledger", return_value=ledger_data):
        summary = get_summary("metrics.jsonl", "experiment_ledger.jsonl")

    assert summary["val_bpb"] == 1.234
    assert summary["tok_per_sec"] == 1100
    assert "loss" in summary
