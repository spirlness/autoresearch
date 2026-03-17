from __future__ import annotations

import gc
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .compile import (
    AVAILABLE_INDUCTOR_MODES,
    maybe_compile_function,
    prepare_compile_environment,
    resolve_compile_backend,
    resolve_optimizer_compile_backend,
    validate_compile_backend,
    validate_compile_mode,
)
from .config import (
    HEAD_DIM,
    build_runtime_config,
    parse_args,
)
from .model import (
    GPT,
    build_model_config,
    compute_mfu,
    estimate_device_peak_flops,
    norm,
    resolve_attention_backend,
)

if TYPE_CHECKING:
    from prepare import Tokenizer


H100_BF16_PEAK_FLOPS = 989.5e12
RESULTS_DIR = Path("results")
LOG_DIR = RESULTS_DIR / "logs"
METRICS_PATH = RESULTS_DIR / "metrics.jsonl"
EXPERIMENT_LEDGER_PATH = RESULTS_DIR / "experiment_ledger.jsonl"
RESEARCH_LOOP_STATE_DIR = RESULTS_DIR / "research_loop"


def ensure_results_layout() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingState:
    step: int = 0
    tokens_processed: int = 0
    t_start: float = 0.0
    t_start_training: float = 0.0
    steady_training_time: float = 0.0
    end_to_end_training_time: float = 0.0
    measured_steps: int = 0
    lossf_mean: float | None = None
    last_dt: float = 0.0

    @property
    def elapsed_training_time(self) -> float:
        if self.t_start_training == 0:
            return 0.0
        return time.time() - self.t_start_training


class Trainer:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.state = TrainingState(t_start=time.time())

        # Hardware & Model Configuration
        self.device = "cuda"
        self.device_props = torch.cuda.get_device_properties(self.device)
        self.device_peak_flops = estimate_device_peak_flops(self.device_props)
        self.autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

        # Resolve attention op (must be passed to GPT)
        attention_res = resolve_attention_backend()
        self.attention_name = (
            attention_res[0] if isinstance(attention_res, tuple) else attention_res
        )
        self.attention_op = (
            attention_res[1] if isinstance(attention_res, tuple) else None
        )

        # Model setup
        # Note: ModelSettings object now has vocab_size attribute from previous fix
        self.model_config = build_model_config(
            depth=runtime.model.depth,
            max_seq_len=runtime.model.max_seq_len,
            vocab_size=runtime.model.vocab_size,
            aspect_ratio=runtime.model.aspect_ratio,
            head_dim=HEAD_DIM,
            window_pattern=runtime.model.window_pattern,
            activation_checkpoint=runtime.model.activation_checkpoint,
            ve_gate_channels=runtime.model.ve_gate_channels,
            softcap=runtime.model.softcap,
        )
        self.raw_model = GPT(self.model_config, attention_op=self.attention_op).to(
            self.device
        )
        self.raw_model.init_weights()
        self.num_params = sum(p.numel() for p in self.raw_model.parameters())
        # Keep MFU reporting aligned with the historical benchmark tables by
        # using the model-level FLOP estimate instead of a raw parameter count.
        self.num_flops_per_token = self.raw_model.estimate_flops()

        # Optimizer & Scheduler
        self.optimizer = self.raw_model.setup_optimizer(
            unembedding_lr=runtime.optimization.unembedding_lr,
            embedding_lr=runtime.optimization.embedding_lr,
            matrix_lr=runtime.optimization.matrix_lr,
            weight_decay=runtime.optimization.weight_decay,
            adam_betas=runtime.optimization.adam_betas,
            scalar_lr=runtime.optimization.scalar_lr,
            optimizer_compile_backend=runtime.compile.optimizer_backend,
            compile_mode=runtime.compile.mode,
        )
        self.lr_func = self._build_lr_scheduler()

        # Compilation & Execution Functions
        # prepare_compile_environment already handles platform setup via utils/platform
        prepare_compile_environment(
            model_backend=runtime.compile.model_backend,
            optimizer_backend=runtime.compile.optimizer_backend,
        )
        (
            self.model_for_execution,
            self.model_for_eval,
            self.microstep_fn,
            self.trunk_fn,
        ) = self._build_execution_functions()

        # Actual tokens processed per optimizer step (used for tok/sec and MFU)
        # Computed here so _log_metrics / _report_final_stats can rely on it
        # without getattr fallbacks.
        self.actual_total_batch_size = (
            runtime.model.device_batch_size
            * runtime.model.max_seq_len
            * max(runtime.grad_accum_steps_override, 1)
        )

    def _aggregate_rate_metrics(self, completed_steps: int):
        total_batch_size = self.actual_total_batch_size
        instant_tok_per_sec = (
            total_batch_size / self.state.last_dt if self.state.last_dt > 0 else None
        )
        instant_mfu = compute_mfu(
            self.device_peak_flops, self.num_flops_per_token, instant_tok_per_sec
        )
        warmup_tok_per_sec = (
            total_batch_size
            * self.state.measured_steps
            / self.state.steady_training_time
            if self.state.measured_steps > 0 and self.state.steady_training_time > 0
            else None
        )
        warmup_mfu = compute_mfu(
            self.device_peak_flops, self.num_flops_per_token, warmup_tok_per_sec
        )
        end_to_end_tok_per_sec = (
            total_batch_size * completed_steps / self.state.end_to_end_training_time
            if completed_steps > 0 and self.state.end_to_end_training_time > 0
            else None
        )
        end_to_end_mfu = compute_mfu(
            self.device_peak_flops, self.num_flops_per_token, end_to_end_tok_per_sec
        )
        return {
            "instant": {"tok_per_sec": instant_tok_per_sec, "mfu": instant_mfu},
            "warmup_excluded": {
                "tok_per_sec": warmup_tok_per_sec,
                "mfu": warmup_mfu,
                "steps": self.state.measured_steps,
                "seconds": self.state.steady_training_time,
            },
            "end_to_end": {
                "tok_per_sec": end_to_end_tok_per_sec,
                "mfu": end_to_end_mfu,
                "steps": completed_steps,
                "seconds": self.state.end_to_end_training_time,
            },
        }

    def _build_lr_scheduler(self):
        opt = self.runtime.optimization

        def lr_func(step, total_steps):
            warmup_steps = int(opt.warmup_ratio * total_steps)
            warmdown_steps = int(opt.warmdown_ratio * total_steps)

            if step < warmup_steps:
                return step / warmup_steps if warmup_steps > 0 else 1.0
            if step > total_steps - warmdown_steps:
                cooldown_progress = (
                    step - (total_steps - warmdown_steps)
                ) / warmdown_steps
                cooldown = 0.5 * (1 + math.cos(math.pi * cooldown_progress))
                return cooldown + (1 - cooldown) * opt.final_lr_frac
            return 1.0

        return lr_func

    def _get_muon_momentum(self, step):
        opt = self.runtime.optimization
        frac = (
            min(step / opt.muon_warmup_steps, 1) if opt.muon_warmup_steps > 0 else 1.0
        )
        return (1 - frac) * 0.85 + frac * 0.95

    def _build_execution_functions(self):
        runtime = self.runtime
        raw_model = self.raw_model
        autocast_ctx = self.autocast_ctx
        grad_accum_steps = self.runtime.grad_accum_steps_override

        model_for_execution = raw_model
        if runtime.compile.use_compiled_model:
            model_for_execution = maybe_compile_function(
                raw_model,
                backend=runtime.compile.model_backend,
                compile_mode=runtime.compile.mode,
                dynamic=False,
            )

        def run_trunk_forward(x, idx, cos, sin):
            return raw_model.forward_trunk(x, idx, cos, sin)

        trunk_fn = run_trunk_forward
        if runtime.compile.use_compiled_trunk:
            trunk_fn = maybe_compile_function(
                run_trunk_forward,
                backend=runtime.compile.model_backend,
                compile_mode=runtime.compile.mode,
                dynamic=False,
            )

        def run_forward(idx, targets, reduction: str = "mean"):
            if runtime.compile.use_compiled_trunk:
                seq_len = idx.size(1)
                cos = raw_model.cos[:, :seq_len]
                sin = raw_model.sin[:, :seq_len]
                x = raw_model.transformer.wte(idx)
                x = norm(x)
                x = trunk_fn(x, idx, cos, sin)
                return raw_model.compute_loss(x, targets=targets, reduction=reduction)
            return model_for_execution(idx, targets, reduction=reduction)

        def run_microstep(x, y):
            with autocast_ctx:
                loss = run_forward(x, y)
            (loss / grad_accum_steps).backward()
            return loss.detach()

        microstep_fn = run_microstep
        if runtime.compile.use_compiled_microstep:
            microstep_fn = maybe_compile_function(
                run_microstep,
                backend=runtime.compile.model_backend,
                compile_mode=runtime.compile.mode,
                dynamic=False,
            )

        model_for_eval = (
            model_for_execution if runtime.compile.use_compiled_model else raw_model
        )
        return model_for_execution, model_for_eval, microstep_fn, trunk_fn

    def _log_metrics(self, epoch: int):
        runtime = self.runtime
        state = self.state
        completed_steps = state.step + 1
        metrics_groups = self._aggregate_rate_metrics(completed_steps)
        instant = metrics_groups["instant"]
        warmup_excluded = metrics_groups["warmup_excluded"]
        end_to_end = metrics_groups["end_to_end"]

        # Format strings
        inst_tok_per_sec_str = (
            f"{instant['tok_per_sec']:,.0f}"
            if instant["tok_per_sec"] is not None
            else "n/a"
        )
        warmup_tok_per_sec_str = (
            f"{warmup_excluded['tok_per_sec']:,.0f}"
            if warmup_excluded["tok_per_sec"] is not None
            else "warming"
        )
        end_to_end_tok_per_sec_str = (
            f"{end_to_end['tok_per_sec']:,.0f}"
            if end_to_end["tok_per_sec"] is not None
            else "n/a"
        )
        warmup_mfu_str = (
            f"{warmup_excluded['mfu']:.1f}%"
            if warmup_excluded["mfu"] is not None
            else "warming"
        )
        end_to_end_mfu_str = (
            f"{end_to_end['mfu']:.1f}%" if end_to_end["mfu"] is not None else "n/a"
        )

        if runtime.benchmark.enabled:
            remaining_str = f"{max(runtime.benchmark.steps - completed_steps, 0)} steps"
            progress_str = f"{100 * completed_steps / runtime.benchmark.steps:.1f}%"
        else:
            remaining_str = (
                f"{max(0.0, runtime.time_budget - state.elapsed_training_time):.0f}s"
            )
            progress_str = (
                f"{100 * state.elapsed_training_time / runtime.time_budget:.1f}%"
            )

        loss_str = f"{state.lossf_mean:.6f}" if state.lossf_mean is not None else "n/a"
        print(
            f"step {state.step:05d} ({progress_str}) | loss: {loss_str} | dt: {state.last_dt * 1000:.0f}ms | "
            f"inst tok/sec: {inst_tok_per_sec_str} | warm tok/sec: {warmup_tok_per_sec_str} | "
            f"e2e tok/sec: {end_to_end_tok_per_sec_str} | warm mfu: {warmup_mfu_str} | "
            f"e2e mfu: {end_to_end_mfu_str} | "
            f"epoch: {epoch} | remaining: {remaining_str}",
            flush=True,
        )

        # Write to structured log file
        metrics = {
            "step": state.step,
            "progress": progress_str,
            "loss": state.lossf_mean,
            "dt_ms": state.last_dt * 1000,
            "epoch": epoch,
            "timestamp": time.time(),
            "instant": {
                "tok_per_sec": instant["tok_per_sec"],
                "mfu": instant["mfu"],
                "h100_mfu": compute_mfu(
                    H100_BF16_PEAK_FLOPS,
                    self.num_flops_per_token,
                    instant["tok_per_sec"],
                ),
            },
            "warmup_excluded": warmup_excluded,
            "end_to_end": end_to_end,
        }
        ensure_results_layout()
        with METRICS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def _update_optimizer_params(self):
        state = self.state
        runtime = self.runtime
        total_steps_est = (
            runtime.benchmark.steps if runtime.benchmark.enabled else 1000000
        )
        lrm = self.lr_func(state.step, total_steps_est)
        muon_momentum = self._get_muon_momentum(state.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * lrm
            if param_group["kind"] == "muon":
                param_group["momentum"] = muon_momentum

    def _run_training_step(self, train_loader, grad_accum_steps):
        self.optimizer.zero_grad(set_to_none=True)
        loss = None
        epoch = 0
        for _ in range(grad_accum_steps):
            x, y, epoch = next(train_loader)
            loss = self.microstep_fn(x, y)
        self.optimizer.step()
        return loss, epoch

    def _finalize_step(self, dt, loss_val, epoch, actual_total_batch_size):
        state = self.state
        runtime = self.runtime

        state.last_dt = dt
        state.end_to_end_training_time += dt

        if state.lossf_mean is None:
            state.lossf_mean = loss_val
        else:
            state.lossf_mean = state.lossf_mean * 0.95 + loss_val * 0.05

        is_benchmark_warmup = (
            runtime.benchmark.enabled
            and (state.step < runtime.benchmark.steps - 1)
            and (state.step < runtime.benchmark.warmup_steps)
        )
        if not is_benchmark_warmup:
            state.steady_training_time += dt
            state.measured_steps += 1

        should_log = (
            state.step == 0
            or is_benchmark_warmup
            or ((state.step + 1) % runtime.benchmark.log_interval == 0)
            or (runtime.benchmark.enabled and state.step + 1 >= runtime.benchmark.steps)
        )

        if should_log:
            self._log_metrics(epoch)

        if state.step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (state.step + 1) % 1000 == 0:
            gc.collect()

        state.step += 1
        state.tokens_processed = state.step * actual_total_batch_size

    def train(self, tokenizer: "Tokenizer", train_loader: Any):
        runtime = self.runtime
        state = self.state
        self.raw_model.train()
        if self.model_for_execution is not self.raw_model and hasattr(
            self.model_for_execution, "train"
        ):
            self.model_for_execution.train()

        # actual_total_batch_size is already computed in __init__; just use it.
        actual_total_batch_size = self.actual_total_batch_size
        grad_accum_steps = runtime.grad_accum_steps_override

        # Reset structured metrics for the current train run.
        ensure_results_layout()
        METRICS_PATH.write_text("", encoding="utf-8")

        print(f"Vocab size: {tokenizer.get_vocab_size():,}")
        print(f"Attention backend: {self.attention_name}")
        print(_compile_status(runtime))
        print(
            f"Time budget: {runtime.time_budget}s"
            if not runtime.benchmark.enabled
            else f"Benchmark steps: {runtime.benchmark.steps}"
        )

        state.t_start_training = time.time()

        while True:
            t_step_start = time.time()

            self._update_optimizer_params()
            loss, epoch = self._run_training_step(train_loader, grad_accum_steps)

            torch.cuda.synchronize()
            t_step_end = time.time()
            dt = t_step_end - t_step_start

            loss_val = loss.item() if loss is not None else 0.0
            self._finalize_step(dt, loss_val, epoch, actual_total_batch_size)

            # Termination conditions
            if runtime.benchmark.enabled and state.step >= runtime.benchmark.steps:
                break
            if (
                not runtime.benchmark.enabled
                and state.elapsed_training_time >= runtime.time_budget
            ):
                break

        print("\nTraining completed.")
        return state


def _compile_status(runtime):
    if runtime.compile.model_backend == "off":
        return "torch.compile: disabled"
    if runtime.compile.model_backend == "inductor":
        return f"torch.compile: enabled ({runtime.compile.model_backend}, mode={runtime.compile.mode}, scope={runtime.compile.scope})"
    return f"torch.compile: enabled ({runtime.compile.model_backend}, scope={runtime.compile.scope})"


from .analyzer import build_research_progress_report, find_best_result, get_summary
from .mutator import suggest_research_env_vars
from .orchestrator import run_experiment


def build_research_loop_extra_args(args: Any) -> list[str]:
    return [
        "--benchmark-steps",
        str(args.benchmark_steps),
        "--compile-backend",
        args.compile_backend,
        "--compile-mode",
        args.compile_mode,
        "--compile-scope",
        args.compile_scope,
        "--optimizer-compile-backend",
        args.optimizer_compile_backend,
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--seed",
        str(args.seed),
    ]


def compute_oom_recovery_settings(
    device_batch_size: int, grad_accum_steps: int
) -> tuple[int, int, int, int]:
    new_batch_size = max(1, device_batch_size // 2)
    target_effective_batch = device_batch_size * max(grad_accum_steps, 1)
    new_grad_accum = max(1, math.ceil(target_effective_batch / new_batch_size))
    recovered_effective_batch = new_batch_size * new_grad_accum
    return (
        new_batch_size,
        new_grad_accum,
        target_effective_batch,
        recovered_effective_batch,
    )


def _format_env_lines(env_vars: dict[str, str]) -> list[str]:
    if not env_vars:
        return ["- none"]
    return [f"- `{key}={value}`" for key, value in sorted(env_vars.items())]


def render_next_research_run_markdown(
    progress: dict[str, Any], next_env_vars: dict[str, str]
) -> str:
    lines = ["# Next Research Run", ""]
    lines.append(f"Latest iteration: {progress.get('latest_iteration')}")
    lines.append(f"Best iteration: {progress.get('best_iteration')}")
    lines.append(
        f"Latest run improved frontier: {progress.get('current_is_best')}"
    )
    lines.append("")
    lines.append("## Best Summary")
    best_summary = progress.get("best_summary", {})
    if best_summary:
        for key, value in sorted(best_summary.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Best Env Vars")
    lines.extend(_format_env_lines(progress.get("best_env_vars", {})))
    lines.append("")
    lines.append("## Recommended Next Env Vars")
    lines.extend(_format_env_lines(next_env_vars))
    lines.append("")
    return "\n".join(lines)


def persist_research_artifacts(
    results: list[dict[str, Any]], *, state_dir: str | Path = RESEARCH_LOOP_STATE_DIR
) -> dict[str, str]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    history_path = state_dir / "history.json"
    best_run_path = state_dir / "best_run.json"
    next_env_path = state_dir / "next_run_env.json"
    next_run_markdown_path = state_dir / "NEXT_RUN.md"

    progress = build_research_progress_report(results)
    best_result = find_best_result(results)
    latest_result = results[-1] if results else {}
    next_env_vars = latest_result.get("recommended_next_env_vars", {})
    timestamp = time.time()

    history_path.write_text(
        json.dumps(
            {"updated_at": timestamp, "iterations": results},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    best_run_path.write_text(
        json.dumps(
            {
                "updated_at": timestamp,
                "progress": progress,
                "best_result": best_result,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    next_env_path.write_text(
        json.dumps(
            {
                "updated_at": timestamp,
                "best_iteration": progress.get("best_iteration"),
                "recommended_next_env_vars": next_env_vars,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    next_run_markdown_path.write_text(
        render_next_research_run_markdown(progress, next_env_vars),
        encoding="utf-8",
    )

    return {
        "history_path": str(history_path),
        "best_run_path": str(best_run_path),
        "next_env_path": str(next_env_path),
        "next_run_markdown_path": str(next_run_markdown_path),
    }


def run_research_loop(
    iterations: int = 3,
    timeout: int | None = None,
    profile: str = "baseline",
    extra_args: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute the end-to-end research loop."""
    results = []

    # We maintain a dict of env vars that gets updated across iterations
    current_env_vars = {}

    for i in range(iterations):
        print(f"\n--- Research Iteration {i+1}/{iterations} ---")
        if current_env_vars:
            print(f"Current overrides: {current_env_vars}")

        # 1. Run Experiment
        experiment_res = run_experiment(
            timeout=timeout,
            profile=profile,
            extra_args=extra_args,
            env_vars=current_env_vars.copy(),
        )

        # 2. Analyze Results
        summary = {}
        if experiment_res.get("status") == "success":
            summary = get_summary(str(METRICS_PATH), str(EXPERIMENT_LEDGER_PATH))
        else:
            print(f"Iteration {i+1} experiment did not succeed: {experiment_res.get('status')}")

        # Record iteration result
        iteration_result = {
            "iteration": i + 1,
            "experiment": experiment_res,
            "summary": summary,
            "applied_env_vars": current_env_vars.copy(),
        }
        results.append(iteration_result)

        print(f"Iteration {i+1} summary: {summary}")
        progress = build_research_progress_report(results)
        iteration_result["progress"] = progress
        if progress.get("current_is_best"):
            print(f"Iteration {i+1} established a new best run.")
        elif progress.get("best_iteration") is not None:
            print(
                f"Best run remains iteration {progress['best_iteration']} "
                f"with summary {progress.get('best_summary', {})}"
            )

        recommended_next_env_vars = suggest_research_env_vars(results)
        iteration_result["recommended_next_env_vars"] = recommended_next_env_vars
        iteration_result["artifact_paths"] = persist_research_artifacts(results)

        # 3. Mutate (if not last iteration)
        if i < iterations - 1:
            current_env_vars = dict(recommended_next_env_vars)
            print(f"Prepared env overrides for next iteration: {current_env_vars}")

    return results

def main() -> int:
    args = parse_args(AVAILABLE_INDUCTOR_MODES)

    # Check for research loop trigger
    if hasattr(args, "research_iterations") and args.research_iterations > 0:
        run_research_loop(
            iterations=args.research_iterations,
            timeout=args.research_timeout or None,
            profile=args.experiment_profile,
            extra_args=build_research_loop_extra_args(args),
        )
        return 0
    model_compile_backend = resolve_compile_backend(args.compile_backend)
    validate_compile_backend(model_compile_backend)
    validate_compile_mode(model_compile_backend, args.compile_mode)
    optimizer_compile_backend = resolve_optimizer_compile_backend(
        args.optimizer_compile_backend, model_compile_backend
    )

    # Prepare Data and Tokenizer
    from prepare import Tokenizer, evaluate_bpb
    from .token_cache import ensure_train_token_cache, make_token_window_loader

    tokenizer = Tokenizer.from_directory()

    runtime = build_runtime_config(
        args,
        model_compile_backend=model_compile_backend,
        optimizer_compile_backend=optimizer_compile_backend,
        vocab_size=tokenizer.get_vocab_size(),
    )
    set_random_seed(runtime.seed)

    import dataclasses

    train_cache = ensure_train_token_cache(tokenizer)
    cache_status = "built" if train_cache.built else "using"
    print(
        f"Train token cache: {cache_status} {train_cache.cache_path} "
        f"({train_cache.num_tokens:,} tokens)",
        flush=True,
    )

    max_retries = 3
    state = None
    trainer = None

    # Robust OOM Handling Loop
    for attempt in range(max_retries + 1):
        try:
            trainer = Trainer(runtime)
            train_loader = make_token_window_loader(
                train_cache,
                runtime.model.device_batch_size,
                runtime.model.max_seq_len,
                device=trainer.device,
                seed=runtime.seed + 1,
            )
            state = trainer.train(tokenizer, train_loader)
            break
        except torch.cuda.OutOfMemoryError as e:
            if attempt == max_retries or runtime.model.device_batch_size <= 1:
                print(
                    "\n[System] OOM: Exhausted retries or reached min batch size. Hardware limit."
                )
                with METRICS_PATH.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps({"failure_reason": "OOM", "message": str(e)}) + "\n"
                    )
                return 1

            print(
                f"\n[System] OOM caught. Halving device_batch_size. Retry {attempt + 1}/{max_retries}"
            )
            torch.cuda.empty_cache()

            # Keep the recovered effective batch size as close as possible to the
            # original target. Exact preservation is impossible for odd sizes.
            (
                new_batch_size,
                new_grad_accum,
                target_effective_batch,
                recovered_effective_batch,
            ) = compute_oom_recovery_settings(
                runtime.model.device_batch_size,
                runtime.grad_accum_steps_override,
            )

            print(
                "[System] Adjusted config: "
                f"device_batch_size={new_batch_size}, "
                f"grad_accum_steps={new_grad_accum}, "
                f"effective_batch={recovered_effective_batch} "
                f"(target={target_effective_batch})"
            )
            ensure_results_layout()
            with EXPERIMENT_LEDGER_PATH.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "OOM_RECOVERY",
                            "new_device_batch_size": new_batch_size,
                            "new_grad_accum_steps": new_grad_accum,
                            "target_effective_batch": target_effective_batch,
                            "recovered_effective_batch": recovered_effective_batch,
                        }
                    )
                    + "\n"
                )

            new_model = dataclasses.replace(
                runtime.model, device_batch_size=new_batch_size
            )
            runtime = dataclasses.replace(
                runtime, model=new_model, grad_accum_steps_override=new_grad_accum
            )

    if state is None:
        return 1

    # Final Eval & Stats
    val_bpb = None
    val_bpb_std = None
    if not runtime.benchmark.enabled:
        trainer.raw_model.eval()
        trainer.model_for_eval.eval()
        with trainer.autocast_ctx:
            val_bpb, val_bpb_std = evaluate_bpb(
                trainer.model_for_eval, tokenizer, runtime.model.device_batch_size
            )

    # Report results
    _report_final_stats(trainer, state, val_bpb, val_bpb_std)
    return 0


def _report_final_stats(trainer, state, val_bpb, val_bpb_std=None):
    runtime = trainer.runtime
    t_end = time.time()
    total_batch_size = trainer.actual_total_batch_size
    end_to_end_tok_per_sec = (
        total_batch_size * state.step / state.end_to_end_training_time
        if state.end_to_end_training_time > 0 and state.step > 0
        else None
    )
    warmup_excluded_tok_per_sec = (
        total_batch_size * state.measured_steps / state.steady_training_time
        if state.steady_training_time > 0 and state.measured_steps > 0
        else None
    )
    end_to_end_mfu = compute_mfu(
        trainer.device_peak_flops, trainer.num_flops_per_token, end_to_end_tok_per_sec
    )
    warmup_excluded_mfu = compute_mfu(
        trainer.device_peak_flops,
        trainer.num_flops_per_token,
        warmup_excluded_tok_per_sec,
    )
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("\n---")
    print(f"mode:             {'benchmark' if runtime.benchmark.enabled else 'train'}")
    print(f"profile:          {runtime.experiment_profile}")
    if val_bpb is not None:
        if val_bpb_std is not None:
            print(f"val_bpb:          {val_bpb:.6f} ± {val_bpb_std:.6f}")
        else:
            print(f"val_bpb:          {val_bpb:.6f}")
    else:
        print("val_bpb:          skipped")
    print(f"total_seconds:    {t_end - state.t_start:.1f}")
    print(
        f"warmup_excluded_tok_per_sec: {warmup_excluded_tok_per_sec:,.0f}"
        if warmup_excluded_tok_per_sec is not None
        else "warmup_excluded_tok_per_sec: n/a"
    )
    print(
        f"warmup_excluded_mfu_percent: {warmup_excluded_mfu:.2f}"
        if warmup_excluded_mfu is not None
        else "warmup_excluded_mfu_percent: n/a"
    )
    print(
        f"end_to_end_tok_per_sec: {end_to_end_tok_per_sec:,.0f}"
        if end_to_end_tok_per_sec is not None
        else "end_to_end_tok_per_sec: n/a"
    )
    print(
        f"end_to_end_mfu_percent: {end_to_end_mfu:.2f}"
        if end_to_end_mfu is not None
        else "end_to_end_mfu_percent: n/a"
    )
    print(f"total_tokens_M:   {state.step * total_batch_size / 1e6:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print("---")

    # Record the experiment in the ledger for agent introspection
    ledger_entry = {
        "timestamp": time.time(),
        "profile": runtime.experiment_profile,
        "val_bpb": val_bpb,
        "val_bpb_std": val_bpb_std,
        "total_seconds": t_end - state.t_start,
        "warmup_excluded_tok_per_sec": warmup_excluded_tok_per_sec,
        "warmup_excluded_mfu": warmup_excluded_mfu,
        "end_to_end_tok_per_sec": end_to_end_tok_per_sec,
        "end_to_end_mfu": end_to_end_mfu,
        "peak_vram_mb": peak_vram_mb,
        "config": {
            "depth": runtime.model.depth,
            "max_seq_len": runtime.model.max_seq_len,
            "device_batch_size": runtime.model.device_batch_size,
            "grad_accum_steps": runtime.grad_accum_steps_override,
            "aspect_ratio": runtime.model.aspect_ratio,
        },
    }
    ensure_results_layout()
    with EXPERIMENT_LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
