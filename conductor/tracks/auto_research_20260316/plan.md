# Implementation Plan: End-to-End Autonomous Research Loop

## Phase 1: Results Analyzer & Metrics Parsing
- [ ] Task: Write Tests for `analyzer.py` (parse `metrics.jsonl`, extract `val_bpb`)
- [ ] Task: Implement `autoresearch_trainer/analyzer.py`
- [ ] Task: Conductor - User Manual Verification 'Results Analyzer & Metrics Parsing' (Protocol in workflow.md)

## Phase 2: Experiment Orchestrator
- [ ] Task: Write Tests for `orchestrator.py` (launching processes, timeout handling)
- [ ] Task: Implement `autoresearch_trainer/orchestrator.py`
- [ ] Task: Conductor - User Manual Verification 'Experiment Orchestrator' (Protocol in workflow.md)

## Phase 3: Code/Config Mutator
- [ ] Task: Write Tests for `mutator.py` (modifying `config.py` and code files)
- [ ] Task: Implement `autoresearch_trainer/mutator.py`
- [ ] Task: Conductor - User Manual Verification 'Code/Config Mutator' (Protocol in workflow.md)

## Phase 4: End-to-End Loop Integration
- [ ] Task: Write Tests for `runner.py` research loop coordination
- [ ] Task: Implement integrated research loop in `autoresearch_trainer/runner.py`
- [ ] Task: Conductor - User Manual Verification 'End-to-End Loop Integration' (Protocol in workflow.md)

## Phase 5: Verification & Success Demonstration
- [ ] Task: Conduct a 3-experiment run and verify results
- [ ] Task: Conductor - User Manual Verification 'Verification & Success Demonstration' (Protocol in workflow.md)
