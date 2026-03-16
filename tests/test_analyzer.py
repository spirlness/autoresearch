import json
from autoresearch_trainer.analyzer import parse_metrics, parse_ledger, get_summary

def test_parse_metrics(tmp_path):
    metrics_file = tmp_path / "metrics.jsonl"
    data = [
        {"step": 0, "loss": 10.0, "end_to_end": {"tok_per_sec": 1000, "mfu": 10.0}},
        {"step": 1, "loss": 9.0, "end_to_end": {"tok_per_sec": 1100, "mfu": 11.0}},
    ]
    with open(metrics_file, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    parsed = parse_metrics(str(metrics_file))
    assert len(parsed) == 2
    assert parsed[-1]["loss"] == 9.0
    assert parsed[-1]["end_to_end"]["tok_per_sec"] == 1100

def test_parse_ledger(tmp_path):
    ledger_file = tmp_path / "experiment_ledger.jsonl"
    data = [
        {"val_bpb": 1.234, "end_to_end_tok_per_sec": 1100, "peak_vram_mb": 5000},
    ]
    with open(ledger_file, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    parsed = parse_ledger(str(ledger_file))
    assert len(parsed) == 1
    assert parsed[0]["val_bpb"] == 1.234

def test_get_summary(tmp_path):
    metrics_file = tmp_path / "metrics.jsonl"
    ledger_file = tmp_path / "experiment_ledger.jsonl"
    
    metrics_data = [{"step": 1, "loss": 9.0, "end_to_end": {"tok_per_sec": 1100}}]
    ledger_data = [{"val_bpb": 1.234, "end_to_end_tok_per_sec": 1100}]
    
    with open(metrics_file, "w") as f:
        for entry in metrics_data:
            f.write(json.dumps(entry) + "\n")
    with open(ledger_file, "w") as f:
        for entry in ledger_data:
            f.write(json.dumps(entry) + "\n")
            
    summary = get_summary(str(metrics_file), str(ledger_file))
    assert summary["val_bpb"] == 1.234
    assert summary["tok_per_sec"] == 1100
    assert "loss" in summary
