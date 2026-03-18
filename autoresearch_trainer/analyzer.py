import json
import math
import os
import statistics
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


def score_benchmark_summary(summary: Dict[str, Any]) -> tuple[float, float, float]:
    """Lower is better. Prefer warmup MFU, then warmup throughput, then lower VRAM."""
    warmup_mfu = _finite_metric(summary.get("warmup_mfu"))
    warmup_tok_per_sec = _finite_metric(summary.get("warmup_tok_per_sec"))
    peak_vram_mb = _finite_metric(summary.get("peak_vram_mb"))
    return (
        -(warmup_mfu if warmup_mfu is not None else 0.0),
        -(warmup_tok_per_sec if warmup_tok_per_sec is not None else 0.0),
        peak_vram_mb if peak_vram_mb is not None else float("inf"),
    )


def aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate repeated run summaries with medians to reduce outlier influence."""
    if not summaries:
        return {}

    aggregate: Dict[str, Any] = {"attempt_count": len(summaries)}
    numeric_keys = {
        key
        for summary in summaries
        for key, value in summary.items()
        if _finite_metric(value) is not None
    }
    for key in sorted(numeric_keys):
        values = [
            finite_value
            for summary in summaries
            if (finite_value := _finite_metric(summary.get(key))) is not None
        ]
        if not values:
            continue
        aggregate[key] = statistics.median(values)
        if len(values) > 1:
            aggregate[f"{key}_mean"] = statistics.fmean(values)
            aggregate[f"{key}_min"] = min(values)
            aggregate[f"{key}_max"] = max(values)
            aggregate[f"{key}_spread"] = max(values) - min(values)

    config = next(
        (dict(summary["config"]) for summary in summaries if isinstance(summary.get("config"), dict)),
        None,
    )
    if config is not None:
        aggregate["config"] = config

    return aggregate


def should_promote_benchmark_candidate(
    candidate_summary: Dict[str, Any],
    incumbent_summary: Dict[str, Any] | None,
) -> tuple[bool, str]:
    """Decide whether a short benchmark is good enough to justify a full train run."""
    candidate_tok = _finite_metric(candidate_summary.get("warmup_tok_per_sec"))
    candidate_mfu = _finite_metric(candidate_summary.get("warmup_mfu"))
    candidate_vram = _finite_metric(candidate_summary.get("peak_vram_mb"))

    if candidate_tok is None and candidate_mfu is None:
        return False, "candidate benchmark is missing warmup throughput/MFU metrics"
    if incumbent_summary is None:
        return True, "no incumbent benchmark; establish a baseline train run"

    incumbent_tok = _finite_metric(incumbent_summary.get("warmup_tok_per_sec"))
    incumbent_mfu = _finite_metric(incumbent_summary.get("warmup_mfu"))
    incumbent_vram = _finite_metric(incumbent_summary.get("peak_vram_mb"))

    tok_ratio = (
        candidate_tok / incumbent_tok
        if candidate_tok is not None and incumbent_tok not in (None, 0.0)
        else None
    )
    mfu_ratio = (
        candidate_mfu / incumbent_mfu
        if candidate_mfu is not None and incumbent_mfu not in (None, 0.0)
        else None
    )
    vram_ratio = (
        candidate_vram / incumbent_vram
        if candidate_vram is not None and incumbent_vram not in (None, 0.0)
        else None
    )

    tok_pass = tok_ratio is None or tok_ratio >= 0.94
    mfu_pass = mfu_ratio is None or mfu_ratio >= 0.97
    vram_pass = vram_ratio is None or vram_ratio <= 1.15
    score_improved = score_benchmark_summary(candidate_summary) < score_benchmark_summary(
        incumbent_summary
    )

    if vram_pass and (tok_pass or mfu_pass or score_improved):
        return True, (
            "benchmark passed promotion gate "
            f"(tok_ratio={tok_ratio}, mfu_ratio={mfu_ratio}, vram_ratio={vram_ratio})"
        )

    return False, (
        "benchmark rejected by promotion gate "
        f"(tok_ratio={tok_ratio}, mfu_ratio={mfu_ratio}, vram_ratio={vram_ratio})"
    )


def is_stable_improvement(
    candidate_summary: Dict[str, Any],
    incumbent_summary: Dict[str, Any] | None,
    *,
    min_val_bpb_delta: float = 0.003,
) -> bool:
    """Require a challenger to beat the incumbent by more than expected noise."""
    if incumbent_summary is None:
        return True

    candidate_val_bpb = _finite_metric(candidate_summary.get("val_bpb"))
    incumbent_val_bpb = _finite_metric(incumbent_summary.get("val_bpb"))
    if candidate_val_bpb is None or incumbent_val_bpb is None:
        return score_summary(candidate_summary) < score_summary(incumbent_summary)

    candidate_std = _finite_metric(candidate_summary.get("val_bpb_std")) or 0.0
    incumbent_std = _finite_metric(incumbent_summary.get("val_bpb_std")) or 0.0
    required_delta = max(min_val_bpb_delta, candidate_std + incumbent_std)
    return candidate_val_bpb <= incumbent_val_bpb - required_delta


def should_confirm_challenger(
    candidate_summary: Dict[str, Any],
    incumbent_summary: Dict[str, Any] | None,
) -> bool:
    """Repeat promising train runs when the improvement is too small to trust yet."""
    if incumbent_summary is None:
        return False
    if score_summary(candidate_summary) >= score_summary(incumbent_summary):
        return False
    return not is_stable_improvement(candidate_summary, incumbent_summary)


def find_best_result(results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Return the best successful result from a research loop, or None if nothing is usable yet."""
    best_result = None
    best_score = None

    for result in results:
        status = result.get("frontier_status") or result.get("experiment", {}).get("status")
        if status != "success":
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
    report["best_benchmark_summary"] = dict(best_result.get("benchmark_summary", {}))
    report["current_is_best"] = (
        latest_result.get("iteration") == best_result.get("iteration")
        and (latest_result.get("frontier_status") or latest_result.get("experiment", {}).get("status")) == "success"
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
        summary["val_bpb_std"] = last_run.get("val_bpb_std", 0.0)
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


def get_summary_for_run(run_dir: str) -> Dict[str, Any]:
    from .artifacts import resolve_run_artifacts

    artifacts = resolve_run_artifacts(run_dir)
    return get_summary(str(artifacts.metrics_path), str(artifacts.ledger_path))
