import json
import os
from autoresearch_trainer.analyzer import get_summary

# Setup dummy data
metrics_file = "test_metrics.jsonl"
ledger_file = "test_ledger.jsonl"

metrics_data = [{"step": 1, "loss": 9.0, "end_to_end": {"tok_per_sec": 1100}}]
ledger_data = [{"val_bpb": 1.234, "end_to_end_tok_per_sec": 1100, "peak_vram_mb": 5000}]

with open(metrics_file, "w") as f:
    for entry in metrics_data:
        f.write(json.dumps(entry) + "\n")
with open(ledger_file, "w") as f:
    for entry in ledger_data:
        f.write(json.dumps(entry) + "\n")

# Verify summary
summary = get_summary(metrics_file, ledger_file)
print(f"Summary: {summary}")

# Expected output: Summary: {'val_bpb': 1.234, 'tok_per_sec': 1100, 'peak_vram_mb': 5000, 'loss': 9.0, 'step': 1}

# Cleanup
os.remove(metrics_file)
os.remove(ledger_file)
