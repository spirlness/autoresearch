import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")
LOG_DIR = RESULTS_DIR / "logs"
TRAIN_COMMAND = ["uv", "run", "python", "-m", "entrypoints.train"]


def run_experiment(
    timeout: int | None = None,
    profile: str = "baseline",
    extra_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Run the training entrypoint as a subprocess with logs under results/logs/."""

    cmd = list(TRAIN_COMMAND)
    if profile:
        cmd.extend(["--experiment-profile", profile])
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    start_time = time.time()
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if label:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "experiment"
            stdout_path = LOG_DIR / f"{safe_label}_stdout.log"
            stderr_path = LOG_DIR / f"{safe_label}_stderr.log"
        else:
            stdout_path = LOG_DIR / "experiment_stdout.log"
            stderr_path = LOG_DIR / "experiment_stderr.log"
        # We redirect stdout/stderr to files to avoid memory pressure from large logs
        with stdout_path.open("w", encoding="utf-8") as f_out, stderr_path.open(
            "w", encoding="utf-8"
        ) as f_err:

            run_kwargs: dict[str, Any] = {
                "stdout": f_out,
                "stderr": f_err,
                "text": True,
                "env": env,
            }
            if timeout is not None:
                run_kwargs["timeout"] = timeout

            result = subprocess.run(cmd, **run_kwargs)

        elapsed = time.time() - start_time

        # Read the tail of logs if needed, or just return paths
        if result.returncode == 0:
            return {
                "status": "success",
                "returncode": 0,
                "elapsed": elapsed,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        else:
            return {
                "status": "failed",
                "returncode": result.returncode,
                "elapsed": elapsed,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        # In a more advanced setup, we might need to hunt down child processes
        # or use taskkill on Windows to ensure GPU memory is released.
        print("[Warning] Experiment timed out. Attempting to ensure resources are freed.")
        return {
            "status": "timeout",
            "elapsed": elapsed,
            "message": f"Experiment timed out after {timeout} seconds"
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "elapsed": elapsed,
            "message": str(e)
        }
