from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from prepare import Tokenizer, evaluate_bpb, make_dataloader

from .compile import (
    AVAILABLE_INDUCTOR_MODES,
    maybe_compile_function,
    prepare_compile_environment,
    resolve_compile_backend,
    resolve_optimizer_compile_backend,
    validate_compile_backend,
    validate_compile_mode,
)
from .config import build_runtime_config, parse_args, HEAD_DIM
from .model import (
    GPT,
    build_model_config,
    compute_mfu,
    estimate_device_peak_flops,
    norm,
    resolve_attention_backend,
    target_tok_per_sec_for_mfu,
)


H100_BF16_PEAK_FLOPS = 989.5e12


@dataclass
class TrainingState:
    step: int = 0
    tokens_processed: int = 0
    t_start: float = 0.0
    t_start_training: float = 0.0
    steady_training_time: float = 0.0
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
        self.attention_name = attention_res[0] if isinstance(attention_res, tuple) else attention_res
        self.attention_op = attention_res[1] if isinstance(attention_res, tuple) else None

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
            softcap=runtime.model.softcap
        )
        self.raw_model = GPT(self.model_config, attention_op=self.attention_op).to(self.device)
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
        self.model_for_execution, self.model_for_eval, self.microstep_fn, self.trunk_fn = self._build_execution_functions()
        
        # Actual tokens processed per optimizer step (used for tok/sec and MFU)
        # Computed here so _log_metrics / _report_final_stats can rely on it
        # without getattr fallbacks.
        self.actual_total_batch_size = (
            runtime.model.device_batch_size
            * runtime.model.max_seq_len
            * max(runtime.grad_accum_steps_override, 1)
        )

    def _build_lr_scheduler(self):
        opt = self.runtime.optimization
        def lr_func(step, total_steps):
            warmup_steps = int(opt.warmup_ratio * total_steps)
            warmdown_steps = int(opt.warmdown_ratio * total_steps)
            
            if step < warmup_steps:
                return step / warmup_steps if warmup_steps > 0 else 1.0
            if step > total_steps - warmdown_steps:
                cooldown_progress = (step - (total_steps - warmdown_steps)) / warmdown_steps
                cooldown = 0.5 * (1 + math.cos(math.pi * cooldown_progress))
                return cooldown + (1 - cooldown) * opt.final_lr_frac
            return 1.0
        return lr_func

    def _get_muon_momentum(self, step):
        opt = self.runtime.optimization
        frac = min(step / opt.muon_warmup_steps, 1) if opt.muon_warmup_steps > 0 else 1.0
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

        def run_microstep(x, y):
            with autocast_ctx:
                loss = model_for_execution(x, y)
            (loss / grad_accum_steps).backward()
            return loss.detach()

        def run_trunk_forward(x, idx, cos, sin):
            return raw_model.forward_trunk(x, idx, cos, sin)

        microstep_fn = run_microstep
        trunk_fn = run_trunk_forward
        
        if runtime.compile.use_compiled_microstep:
            microstep_fn = maybe_compile_function(
                run_microstep,
                backend=runtime.compile.model_backend,
                compile_mode=runtime.compile.mode,
                dynamic=False,
            )
        elif runtime.compile.use_compiled_trunk:
            trunk_fn = maybe_compile_function(
                run_trunk_forward,
                backend=runtime.compile.model_backend,
                compile_mode=runtime.compile.mode,
                dynamic=False,
            )

        model_for_eval = model_for_execution if runtime.compile.use_compiled_model else raw_model
        return model_for_execution, model_for_eval, microstep_fn, trunk_fn

    def _log_metrics(self, tokenizer: Tokenizer, epoch: int):
        runtime = self.runtime
        state = self.state
        
        total_batch_size = self.actual_total_batch_size
        tok_per_sec = total_batch_size / state.last_dt if state.last_dt > 0 else 0.0
        device_mfu = compute_mfu(self.device_peak_flops, self.num_flops_per_token, tok_per_sec)
        h100_mfu = compute_mfu(H100_BF16_PEAK_FLOPS, self.num_flops_per_token, tok_per_sec)
        
        # Format strings
        tok_per_sec_str = f"{tok_per_sec:,.0f}"
        device_mfu_str = f"{device_mfu:.1f}%"
        h100_mfu_str = f"{h100_mfu:.1f}%" if h100_mfu is not None else "n/a"
        
        if runtime.benchmark.enabled:
            remaining_str = f"{max(runtime.benchmark.steps - (state.step + 1), 0)} steps"
            progress_str = f"{100 * (state.step + 1) / runtime.benchmark.steps:.1f}%"
        else:
            remaining_str = f"{max(0.0, runtime.time_budget - state.elapsed_training_time):.0f}s"
            progress_str = f"{100 * state.elapsed_training_time / runtime.time_budget:.1f}%"

        loss_str = f"{state.lossf_mean:.6f}" if state.lossf_mean is not None else "n/a"
        print(
            f"step {state.step:05d} ({progress_str}) | loss: {loss_str} | dt: {state.last_dt*1000:.0f}ms | "
            f"tok/sec: {tok_per_sec_str} | mfu: {device_mfu_str} | h100_mfu: {h100_mfu_str} | "
            f"epoch: {epoch} | remaining: {remaining_str}",
            flush=True,
        )
        
        # Write to structured log file
        metrics = {
            "step": state.step,
            "progress": progress_str,
            "loss": state.lossf_mean,
            "dt_ms": state.last_dt * 1000,
            "tok_per_sec": tok_per_sec,
            "mfu": device_mfu,
            "h100_mfu": h100_mfu,
            "epoch": epoch,
            "timestamp": time.time()
        }
        with open("metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def train(self, tokenizer: Tokenizer, train_loader: Any):
        runtime = self.runtime
        state = self.state
        
        # actual_total_batch_size is already computed in __init__; just use it.
        actual_total_batch_size = self.actual_total_batch_size
        grad_accum_steps = runtime.grad_accum_steps_override
        
        # Reset log file
        with open("metrics.jsonl", "w") as f: pass
        
        print(f"Vocab size: {tokenizer.get_vocab_size():,}")
        print(f"Attention backend: {self.attention_name}")
        print(_compile_status(runtime))
        print(f"Time budget: {runtime.time_budget}s" if not runtime.benchmark.enabled else f"Benchmark steps: {runtime.benchmark.steps}")

        state.t_start_training = time.time()
        
        while True:
            t_step_start = time.time()
            
            # Learning rate and optimizer parameters
            total_steps_est = runtime.benchmark.steps if runtime.benchmark.enabled else 1000000
            lrm = self.lr_func(state.step, total_steps_est)
            muon_momentum = self._get_muon_momentum(state.step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * lrm
                if param_group["kind"] == "muon":
                    param_group["momentum"] = muon_momentum

            # Training step
            self.optimizer.zero_grad(set_to_none=True)
            loss = None
            for _ in range(grad_accum_steps):
                x, y, epoch = next(train_loader)
                loss = self.microstep_fn(x, y)
            
            self.optimizer.step()
            
            torch.cuda.synchronize()
            t_step_end = time.time()
            dt = t_step_end - t_step_start
            state.last_dt = dt
            
            # Update metrics
            loss_val = loss.item() if loss is not None else 0.0
            if state.lossf_mean is None:
                state.lossf_mean = loss_val
            else:
                state.lossf_mean = state.lossf_mean * 0.95 + loss_val * 0.05
            
            is_benchmark_warmup = runtime.benchmark.enabled and (state.step < runtime.benchmark.steps - 1) and (state.step < runtime.benchmark.warmup_steps)
            if not is_benchmark_warmup:
                state.steady_training_time += dt
                state.measured_steps += 1
            
            # Logging
            should_log = (state.step == 0 or is_benchmark_warmup or 
                        ((state.step + 1) % runtime.benchmark.log_interval == 0) or
                        (runtime.benchmark.enabled and state.step + 1 >= runtime.benchmark.steps))
            
            if should_log:
                self._log_metrics(tokenizer, epoch)

            # GC Management
            if state.step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif (state.step + 1) % 1000 == 0:
                gc.collect()

            state.step += 1
            state.tokens_processed = state.step * actual_total_batch_size

            # Termination conditions
            if runtime.benchmark.enabled and state.step >= runtime.benchmark.steps:
                break
            if not runtime.benchmark.enabled and state.elapsed_training_time >= runtime.time_budget:
                break

        print("\nTraining completed.")
        return state


def _compile_status(runtime):
    if runtime.compile.model_backend == "off":
        return "torch.compile: disabled"
    if runtime.compile.model_backend == "inductor":
        return (f"torch.compile: enabled ({runtime.compile.model_backend}, mode={runtime.compile.mode}, scope={runtime.compile.scope})")
    return f"torch.compile: enabled ({runtime.compile.model_backend}, scope={runtime.compile.scope})"


def main() -> int:
    args = parse_args(AVAILABLE_INDUCTOR_MODES)
    model_compile_backend = resolve_compile_backend(args.compile_backend)
    validate_compile_backend(model_compile_backend)
    validate_compile_mode(model_compile_backend, args.compile_mode)
    optimizer_compile_backend = resolve_optimizer_compile_backend(args.optimizer_compile_backend, model_compile_backend)
    
    # Prepare Data and Tokenizer
    tokenizer = Tokenizer.from_directory()

    runtime = build_runtime_config(
        args,
        model_compile_backend=model_compile_backend,
        optimizer_compile_backend=optimizer_compile_backend,
        vocab_size=tokenizer.get_vocab_size()
    )
    
    # Initialize Trainer
    trainer = Trainer(runtime)
    
    train_loader = make_dataloader(
        tokenizer,
        runtime.model.device_batch_size,
        runtime.model.max_seq_len,
        "train",
    )
    
    # Train
    state = trainer.train(tokenizer, train_loader)
    
    # Final Eval & Stats
    val_bpb = None
    if not runtime.benchmark.enabled:
        trainer.model_for_eval.eval()
        with trainer.autocast_ctx:
            val_bpb = evaluate_bpb(trainer.model_for_eval, tokenizer, runtime.model.device_batch_size)

    # Report results
    _report_final_stats(trainer, state, val_bpb)
    return 0


def _report_final_stats(trainer, state, val_bpb):
    runtime = trainer.runtime
    t_end = time.time()
    total_batch_size = trainer.actual_total_batch_size
    
    steady_tok_per_sec = (total_batch_size * state.measured_steps / state.steady_training_time 
                         if state.steady_training_time > 0 and state.measured_steps > 0 else None)
    
    steady_state_mfu = compute_mfu(trainer.device_peak_flops, trainer.num_flops_per_token, steady_tok_per_sec)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"mode:             {'benchmark' if runtime.benchmark.enabled else 'train'}")
    print(f"profile:          {runtime.experiment_profile}")
    print(f"val_bpb:          {val_bpb:.6f}" if val_bpb is not None else "val_bpb:          skipped")
    print(f"total_seconds:    {t_end - state.t_start:.1f}")
    print(f"steady_tok_per_sec: {steady_tok_per_sec:,.0f}" if steady_tok_per_sec is not None else "tok_per_sec:      n/a")
    print(f"mfu_percent:      {steady_state_mfu:.2f}" if steady_state_mfu is not None else "mfu_percent:      n/a")
    print(f"total_tokens_M:   {state.step * total_batch_size / 1e6:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print("---")
