import os
import sys
import time
from autoresearch_trainer.runner import run_research_loop

# Set a very small time budget for each experiment to speed up demonstration
os.environ["TIME_BUDGET"] = "30"

print("Starting 3-iteration research loop demonstration...")
results = run_research_loop(iterations=3, timeout=300)

print(f"\nDemonstration completed with {len(results)} results.")
for res in results:
    summary = res['summary']
    print(f"Iteration {res['iteration']}:")
    print(f"  Status: {res['experiment']['status']}")
    print(f"  Val BPB: {summary.get('val_bpb')}")
    print(f"  Loss: {summary.get('loss')}")
    print(f"  Throughput: {summary.get('tok_per_sec')}")

if len(results) == 3:
    print("\nSUCCESS: End-to-end research loop successfully orchestrated 3 iterations.")
    
    # Check for mutation proof (EMBEDDING_LR should change)
    from autoresearch_trainer import config
    import importlib
    importlib.reload(config)
    print(f"Final EMBEDDING_LR in config: {config.EMBEDDING_LR}")
else:
    print(f"\nFAILURE: Expected 3 results, got {len(results)}")
