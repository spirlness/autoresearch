import os
import subprocess
import time
from typing import Any


def run_experiment(
    timeout: int | None = None,
    profile: str = "baseline",
    extra_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run train.py as a subprocess, optionally guarded by a wall-clock timeout."""

    cmd = ["uv", "run", "train.py"]
    if profile:
        cmd.extend(["--experiment-profile", profile])
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    start_time = time.time()
    try:
        # We redirect stdout/stderr to files to avoid memory pressure from large logs
        with open("experiment_stdout.log", "w", encoding="utf-8") as f_out, \
             open("experiment_stderr.log", "w", encoding="utf-8") as f_err:

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
                "stdout_path": "experiment_stdout.log",
                "stderr_path": "experiment_stderr.log"
            }
        else:
            return {
                "status": "failed",
                "returncode": result.returncode,
                "elapsed": elapsed,
                "stdout_path": "experiment_stdout.log",
                "stderr_path": "experiment_stderr.log"
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
