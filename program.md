# autoresearch

This repository is for autonomous LLM training research on a single GPU.

Your job is to improve the project while respecting the fixed evaluation harness and the current trainer architecture.

## Mission

Optimize for:

1. Lowest `val_bpb` on the fixed 5-minute training budget.
2. Strong steady-state efficiency (`warmup_excluded_tok_per_sec`, `warmup_excluded_mfu`).
3. Reasonable VRAM usage.
4. Simplicity: small wins with cleaner code are better than tiny wins with ugly complexity.

`val_bpb` is the primary metric. Lower is better.

## Operating Style

Behave like a disciplined overnight researcher:

- stay autonomous, but do not thrash
- prefer evidence over novelty
- keep experiments small and attributable
- use the existing frontier before inventing a new search direction
- preserve momentum: once a promising lane is found, exploit locally before jumping elsewhere

Default posture:

- first exploit the best-known region
- then explore one adjacent variation
- only then consider larger code or architecture changes

## Ground Truth

- `entrypoints/prepare.py` is read-only. Do not modify it.
- `evaluate_bpb` in `entrypoints/prepare.py` is the ground truth metric.
- `entrypoints/train.py` is the canonical thin entrypoint. Most meaningful changes belong in `autoresearch_trainer/`.
- The root `prepare.py` and `train.py` files are compatibility shims, not the main implementation targets.
- The trainer package is modular:
  - `config.py` - profiles, runtime config, env-var overrides
  - `model.py` - architecture and attention path
  - `optimizer.py` - fused optimizer logic
  - `compile.py` - compile backend handling
  - `token_cache.py` - mmap cache and train loader
  - `train_session.py` - single-run training lifecycle, OOM recovery, final evaluation
  - `research_loop.py` - autonomous benchmark/train/confirm loop
  - `runner.py` - package facade and CLI export surface
  - `analyzer.py` - result scoring and progress reporting
  - `mutator.py` - next-step env-var suggestions
  - `orchestrator.py` - subprocess experiment execution

## Current Priors

Assume these are good defaults unless new evidence beats them:

- `baseline` / `default` / `mfu50` is the main profile to optimize for quality.
- `throughput` is the preserved throughput reference profile.
- `MLP-only checkpointing` is currently the preferred memory-saving mode.
- On the current Windows stack, `compile_mode=default` is preferred over `max-autotune`.
- Bigger batch alone is usually not the best lever.
- Longer context, `LLLL`, and moderate batch sizes are often better MFU levers than blindly increasing batch.
- The runner already has OOM recovery. Repeated OOM or paging behavior is still a sign to abandon that idea.

If you are doing performance tuning, read `docs/findings/benchmark-findings.md` early.

## Priority Order

When choosing what to do next, use this priority order:

1. Resume from `results/research_loop/` if there is an existing frontier.
2. Re-run or refresh the best-known baseline if the frontier looks stale or ambiguous.
3. Continue local env-var exploration around the best successful run.
4. Run a 20-step benchmark before any substantial code change.
5. Make code changes only when the current frontier suggests a real architectural or runtime bottleneck.
6. Update findings docs only after a result looks durable enough to guide future sessions.

Bias toward local search over random restarts.

## Session Start

At the start of a fresh autonomous session:

1. Inspect git status and avoid overwriting unrelated user changes.
2. Read the current frontier and resume artifacts if they exist:
   - `results/research_loop/NEXT_RUN.md`
   - `results/research_loop/best_run.json`
   - `results/research_loop/next_run_env.json`
   - `results/research_loop/history.json`
3. Read the in-scope files for context:
   - `README.md`
   - `autoresearch_trainer/config.py`
   - `autoresearch_trainer/train_session.py`
   - `autoresearch_trainer/research_loop.py`
   - `autoresearch_trainer/runner.py`
   - `autoresearch_trainer/analyzer.py`
   - `autoresearch_trainer/mutator.py`
   - `autoresearch_trainer/model.py`
   - `autoresearch_trainer/optimizer.py`
   - `autoresearch_trainer/token_cache.py`
   - `autoresearch_trainer/utils/platform.py`
   - `docs/findings/benchmark-findings.md` when tuning performance
4. Verify cached data and tokenizer exist under `~/.cache/autoresearch/`. If not, ask the human to run `uv run python -m entrypoints.prepare`.
5. Treat `results/research_loop/` plus the JSONL outputs under `results/` as the canonical run state.

If there is no existing frontier for this branch/session, establish a fresh baseline before trying improvements.

## Modes Of Work

Use the lightest-weight mode that matches the task.

### 1. Env-var / profile search

Prefer this mode first when you are exploring hyperparameters or runtime knobs without changing source code.

Use the built-in research loop:

```bash
uv run python -m entrypoints.train --research-iterations 3 --research-timeout 600 --experiment-profile baseline
```

What it does:

- launches repeated training subprocesses
- reads `results/metrics.jsonl` and `results/experiment_ledger.jsonl`
- scores the current best run
- recommends the next env-var overrides
- persists resume artifacts under `results/research_loop/`

After each loop, inspect:

- `results/research_loop/NEXT_RUN.md`
- `results/research_loop/best_run.json`
- `results/research_loop/next_run_env.json`

Prefer continuing from the recommended next env vars unless you have a concrete reason to diverge.

Use this mode for most iterations in a long unattended run.

Good knobs for this mode:

- `EMBEDDING_LR`
- `UNEMBEDDING_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- `WARMUP_RATIO`
- `MUON_WARMUP_STEPS`
- `DEVICE_BATCH_SIZE`
- `DEPTH`
- `ASPECT_RATIO`
- `MAX_SEQ_LEN`
- `WINDOW_PATTERN`
- `ACTIVATION_CHECKPOINT`
- `--grad-accum-steps`

Good local-search patterns:

- keep one knob fixed while sweeping one adjacent value
- if a run improves, stay near it for at least 1-3 more tries
- if a run regresses, try one neighboring correction before abandoning the lane
- if multiple nearby tries fail, return to the current frontier

### 2. Benchmark mode

Use this when comparing throughput, compile settings, MFU, or architectural efficiency before committing to a full 5-minute training run.

Recommended commands:

```bash
uv run python -m entrypoints.train --experiment-profile throughput --benchmark-steps 20
uv run python -m entrypoints.train --experiment-profile baseline --benchmark-steps 20
```

Interpretation rules:

- prefer `warmup_excluded` metrics for steady-state comparisons
- treat 20-step-or-more runs as real benchmark evidence
- do not trust 1-step, 3-step, or 5-step runs as final ranking evidence

Benchmark use cases:

- compare compile scopes or compile modes
- compare attention layouts or activation checkpointing strategies
- test whether a code change is worth promoting to a full train run
- validate that a refactor did not hurt throughput

### 3. Code-change mode

Use this only when the frontier is clearly blocked by source code, architecture, optimizer behavior, compile plumbing, or instrumentation gaps.

When changing code:

1. benchmark first if the change affects runtime behavior
2. run a full baseline-quality train if the benchmark looks viable
3. keep the change only if it improves `val_bpb`, or if it preserves quality while clearly simplifying the code

Do not use code-change mode as the default search strategy.

Prefer code changes only when one of these is true:

- the current best frontier is clearly limited by memory, compile behavior, optimizer behavior, or loader behavior
- the project already contains strong evidence for a promising direction
- a simplification could remove complexity without hurting quality

## Commands

Use shell-appropriate commands for this Windows environment.

### Full training run

```bash
uv run python -m entrypoints.train --experiment-profile baseline
```

### Read key metrics from logs

```powershell
Select-String -Path results/logs/experiment_stdout.log -Pattern "^val_bpb:|^peak_vram_mb:|^warmup_excluded_tok_per_sec:|^warmup_excluded_mfu_percent:"
```

### Inspect a crash

```powershell
Get-Content results/logs/experiment_stderr.log -Tail 80
```

### Read generated research frontier

```powershell
Get-Content results/research_loop/NEXT_RUN.md
Get-Content results/research_loop/next_run_env.json
```

## Decision Policy

When deciding whether to keep a change:

- First priority: lower `val_bpb`
- Second priority: better steady-state throughput / MFU
- Third priority: lower or similar VRAM
- Fourth priority: lower code complexity

Examples:

- better `val_bpb` and similar complexity -> keep
- same `val_bpb` and noticeably simpler code -> keep
- tiny `val_bpb` improvement with messy complexity and worse efficiency -> probably discard
- throughput gain with worse `val_bpb` on full train -> do not call it a win

## Decision Templates

Use these templates to avoid ad hoc decisions.

### Benchmark promotion template

Promote a benchmark idea to a full train run only if at least one is true:

- steady-state `tok/s` improves meaningfully without obvious VRAM blow-up
- steady-state `MFU` improves meaningfully without obvious throughput collapse
- the change enables a larger or better-shaped model configuration worth testing
- the code gets materially simpler while holding benchmark behavior roughly steady

Do not promote when:

- the benchmark only improves end-to-end numbers but worsens warmup-excluded behavior
- the gain is tiny and the code is noticeably more complex
- the run is unstable, near-OOM, or obviously paging

### Full-train keep/discard template

Keep a full-train result when:

- `val_bpb` improves clearly, or
- `val_bpb` is effectively flat but the code is simpler, or
- `val_bpb` is effectively flat and efficiency/VRAM improve enough to unlock better future search

Discard when:

- `val_bpb` is worse
- the run crashes for reasons intrinsic to the idea
- the gain is tiny but complexity cost is high

If a result is extremely close to the current best and the direction still looks promising, one confirming rerun is acceptable before deciding.

### Frontier update template

When a run becomes the new best:

1. inspect `results/research_loop/NEXT_RUN.md`
2. inspect `results/research_loop/best_run.json`
3. continue searching locally around that frontier before exploring elsewhere

## Logging

Use the structured outputs for machine-readable and resume-friendly state:

- `results/metrics.jsonl`
- `results/experiment_ledger.jsonl`
- `results/research_loop/history.json`
- `results/research_loop/best_run.json`
- `results/research_loop/next_run_env.json`
- `results/research_loop/NEXT_RUN.md`

Treat `results/research_loop/` as the primary continuation mechanism for autonomous sessions.

## Stop-Loss Rules

Use explicit stop-loss behavior to avoid wasting overnight cycles.

Abandon a lane and return to the current frontier if any of these happen:

- 2 consecutive crashes from the same core idea
- 3 consecutive non-improving runs in the same narrow neighborhood
- repeated OOM recovery on the same direction
- benchmark wins fail to convert into full-train quality gains
- the change requires growing complexity but keeps missing on quality

Escalate from env-var search to code-change mode only after the local neighborhood stops yielding useful movement.

De-escalate from code-change mode back to env-var search when:

- two code ideas in a row fail
- the architecture is churning but benchmark evidence is weak
- the best next move is clearly just another local hyperparameter refinement

## Git Discipline

- Prefer a dedicated branch for a fresh autonomous run, e.g. `autoresearch/<tag>`.
- Commit only code or documentation worth keeping.
- Never revert unrelated user changes.
- If an experiment underperforms, revert only your own change set and only when it is safe to do so.
- If the worktree contains unrelated edits, avoid destructive resets; use a safer alternative.

## Hard Constraints

You must not:

- modify `entrypoints/prepare.py`
- modify the evaluation metric
- add new dependencies
- treat benchmark-only wins as full-training wins
- stop after one failed attempt if there are still promising directions

## Default Autonomous Loop

Use this default loop unless the human gives a more specific task:

1. Read the current frontier from `results/research_loop/` if present.
2. If needed, establish or refresh the baseline.
3. Prefer env-var search with the built-in research loop first.
4. Run 20-step benchmarks before expensive code changes.
5. Promote only the best benchmark ideas into full train runs.
6. Persist and consult `results/research_loop/NEXT_RUN.md` after every round.
7. Stay near the best frontier until stop-loss rules say to pivot.
8. Update `docs/findings/` only when a finding is durable and worth carrying forward.
9. Keep going until explicitly interrupted.

## Commit Rhythm

Use this commit rhythm:

- do not commit every speculative benchmark tweak
- commit when a code change is worth keeping
- do not commit generated logs or `results/research_loop/`
- if a change is discarded, remove only your own change cleanly

For long autonomous runs, prefer fewer but clearer commits over noisy micro-commits.

## Long-Run Hygiene

Over long sessions:

- keep context compact by relying on `results/research_loop/` instead of rereading raw logs repeatedly
- prefer one hypothesis per experiment
- avoid revisiting the exact same env-var signature unless you are explicitly revalidating
- periodically reread `NEXT_RUN.md` so the search direction stays aligned with the latest frontier
- if a run outcome is ambiguous, resolve it quickly with one confirming test, then move on

## If You Get Stuck

When progress stalls:

- reread `docs/findings/benchmark-findings.md`
- inspect `results/research_loop/history.json` for already-tried regions
- inspect `results/research_loop/best_run.json` for the current best config and summary
- combine near-miss ideas instead of repeating them
- prefer one clear hypothesis per experiment
- switch from code changes back to env-var search if architecture churn is not paying off

Stay autonomous, but stay evidence-driven.
