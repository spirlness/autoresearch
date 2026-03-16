import os
import sys
from autoresearch_trainer.runner import run_research_loop

# We will run 2 iterations with 1 benchmark step each
print("Starting research loop verification...")
results = run_research_loop(iterations=2, timeout=600, extra_args=["--benchmark-steps", "1"])

print(f"\nLoop completed with {len(results)} results.")
for res in results:
    print(f"Iteration {res['iteration']}: Status {res['experiment']['status']}, Summary: {res['summary']}")

if len(results) == 2:
    print("Research loop verification succeeded!")
else:
    print(f"Research loop verification failed (expected 2 results, got {len(results)})")
