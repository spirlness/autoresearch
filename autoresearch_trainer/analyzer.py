import json
import math
import os
from typing import List, Dict, Any


def _finite_metric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


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


def score_summary(summary: Dict[str, Any]) -> tuple[float, float, float, float]:
    """Lower is better. Prefer val_bpb, then loss, then higher throughput, then lower VRAM."""
    val_bpb = _finite_metric(summary.get("val_bpb"))
    loss = _finite_metric(summary.get("loss"))
    tok_per_sec = _finite_metric(summary.get("tok_per_sec"))
    peak_vram_mb = _finite_metric(summary.get("peak_vram_mb"))
    return (
        val_bpb if val_bpb is not None else float("inf"),
        loss if loss is not None else float("inf"),
        -(tok_per_sec if tok_per_sec is not None else 0.0),
        peak_vram_mb if peak_vram_mb is not None else float("inf"),
    )


def find_best_result(results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Return the best successful result from a research loop, or None if nothing is usable yet."""
    best_result = None
    best_score = None

    for result in results:
        if result.get("experiment", {}).get("status") != "success":
            continue
        summary = result.get("summary", {})
        score = score_summary(summary)
        if not any(math.isfinite(metric) for metric in score[:2]):
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_result = result

    return best_result


def build_research_progress_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize the current best run and whether the latest run improved the frontier."""
    if not results:
        return {}

    latest_result = results[-1]
    best_result = find_best_result(results)
    report: Dict[str, Any] = {
        "latest_iteration": latest_result.get("iteration"),
        "latest_status": latest_result.get("experiment", {}).get("status"),
    }
    if best_result is None:
        report["best_iteration"] = None
        report["current_is_best"] = False
        return report

    report["best_iteration"] = best_result.get("iteration")
    report["best_summary"] = dict(best_result.get("summary", {}))
    report["best_env_vars"] = dict(best_result.get("applied_env_vars", {}))
    report["current_is_best"] = (
        latest_result.get("iteration") == best_result.get("iteration")
        and latest_result.get("experiment", {}).get("status") == "success"
    )
    return report


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
        summary["warmup_tok_per_sec"] = last_run.get("warmup_excluded_tok_per_sec", 0.0)
        summary["warmup_mfu"] = last_run.get("warmup_excluded_mfu", 0.0)
        summary["peak_vram_mb"] = last_run.get("peak_vram_mb", 0.0)
        summary["config"] = last_run.get("config", {})
    
    if metrics:
        # Take the last entry from metrics for the final step loss
        last_step = metrics[-1]
        summary["loss"] = last_step.get("loss", float("inf"))
        summary["step"] = last_step.get("step", 0)
        
    return summary
