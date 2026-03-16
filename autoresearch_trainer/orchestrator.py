import subprocess
import time
from typing import Dict, Any

def run_experiment(timeout: int = 300, profile: str = "baseline") -> Dict[str, Any]:
    """Run train.py as a subprocess with a timeout."""
    cmd = ["uv", "run", "train.py"]
    if profile:
        cmd.extend(["--experiment-profile", profile])
    
    start_time = time.time()
    try:
        # We use subprocess.run with timeout
        # capture_output=True to keep console clean, or False to see progress
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return {
                "status": "success",
                "returncode": 0,
                "elapsed": elapsed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            return {
                "status": "failed",
                "returncode": result.returncode,
                "elapsed": elapsed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
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
