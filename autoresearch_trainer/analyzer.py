import json
import os
from typing import List, Dict, Any

def parse_metrics(file_path: str) -> List[Dict[str, Any]]:
    """Parse metrics.jsonl file."""
    metrics = []
    if not os.path.exists(file_path):
        return metrics
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics

def parse_ledger(file_path: str) -> List[Dict[str, Any]]:
    """Parse experiment_ledger.jsonl file."""
    ledger = []
    if not os.path.exists(file_path):
        return ledger
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                ledger.append(json.loads(line))
    return ledger

def get_summary(metrics_path: str, ledger_path: str) -> Dict[str, Any]:
    """Get summary from metrics and ledger files."""
    metrics = parse_metrics(metrics_path)
    ledger = parse_ledger(ledger_path)
    
    summary = {}
    if ledger:
        # Take the last entry from the ledger for final stats
        last_run = ledger[-1]
        summary["val_bpb"] = last_run.get("val_bpb", float("inf"))
        summary["tok_per_sec"] = last_run.get("end_to_end_tok_per_sec", 0.0)
        summary["peak_vram_mb"] = last_run.get("peak_vram_mb", 0.0)
    
    if metrics:
        # Take the last entry from metrics for the final step loss
        last_step = metrics[-1]
        summary["loss"] = last_step.get("loss", float("inf"))
        summary["step"] = last_step.get("step", 0)
        
    return summary
