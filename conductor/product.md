# Initial Concept
Autonomous research setup for LLM training where agents modify code and run 5-minute experiments to optimize performance.

# Product Guide

## Vision
To create an autonomous research platform where AI agents can iterate on LLM training setups, model architectures, and hyperparameters to achieve optimal performance within fixed resource constraints.

## Target Users
- **AI Researchers**: Looking for automated ways to explore the search space of model configurations.
- **ML Hobbyists**: Interested in neural network training, optimization, and automation.
- **Autonomous Agents**: The primary "workers" who program the system via `program.md` and execute experiments.

## Core Goals
1. **Minimize Loss**: Drive down `val_bpb` (validation bits per byte) within a 5-minute wall-clock training budget.
2. **Maximize Throughput**: Optimize the number of training tokens processed per second to maximize the information gained from each experiment.
3. **Architectural Innovation**: Enable agents to propose, implement, and validate novel model architectures and training techniques.

## Key Features
- **Experiment Tracking**: Comprehensive logging and history of experiments to ensure the agent can learn from past successes and failures.
- **Fixed-Time Budgeting**: Strict adherence to a 5-minute training window to ensure fair comparisons between different architectural or hyperparameter choices.
- **Structured Metrics**: Automatic generation of `metrics.jsonl` to facilitate analysis by external swarms or plotting utilities.

## Operational Constraints
- **Fixed Time Budget**: Each training run is capped at 5 minutes to ensure rapid iteration and direct comparability.
- **Single-GPU Focus**: Optimized for execution on a single NVIDIA GPU (e.g., RTX 3060, H100).
- **Reproducibility**: All changes and experiment results must be traceable via logs or Git to ensure consistent progress.
