import os
import sys
from autoresearch_trainer.orchestrator import run_experiment

# We will run a very short benchmark to verify orchestration
print("Starting experiment orchestration verification...")
result = run_experiment(timeout=300, profile="baseline", extra_args=["--benchmark-steps", "1"])

print(f"Status: {result['status']}")
if "elapsed" in result:
    print(f"Elapsed: {result['elapsed']:.2f}s")

if result["status"] == "success":
    print("Orchestration succeeded!")
elif result["status"] == "timeout":
    print("Orchestration timed out (as expected if timeout was set too low, but here 60s should be enough for 1 step)")
else:
    print(f"Orchestration failed with returncode {result.get('returncode')}")
    if "stderr" in result:
        print(f"Error: {result['stderr']}")
