# Specification: End-to-End Autonomous Research Loop

## Overview
Implement an autonomous research loop for optimizing LLM training. This loop will orchestrate experiments, mutate training code/configurations, and analyze results to drive continuous improvement in model performance.

## Functional Requirements
1. **Experiment Orchestrator (Orchestrator)**
   - 自动启动并监控由 `train.py` 执行的训练实验。
   - 严格遵守每个实验 5 分钟的时间限制（根据 `product.md`）。
   - 记录实验元数据（开始时间、状态、日志路径）。
2. **代码修改逻辑 (Code Mutator)**
   - 能够修改训练代码或超参数（例如 `autoresearch_trainer/config.py` 或相关文件）。
   - 基于前一个实验的结果，提出并应用新的变更。
3. **结果分析器 (Analyzer)**
   - 解析 `train.py` 生成s的 `metrics.jsonl` 文件。
   - 提取关键指标，如 `val_bpb` 和吞吐量。
   - 为下一轮循环生成总结摘要。
4. **端到端研究循环 (End-to-End Loop)**
   - 协调整个生命周期：修改 -> 运行 -> 分析 -> 再次修改。
   - 支持配置运行次数（本 Track 目标为至少 3 次）。

## Non-Functional Requirements
- **可靠性 (Reliability)**：能够优雅地处理单个实验失败的情况并尝试继续循环。
- **可追溯性 (Traceability)**：所有的代码修改和实验结果必须有完整的日志记录，以便审计。
- **效率 (Efficiency)**：尽可能减少实验之间的切换开销。

## Acceptance Criteria
- 能够成功连续运行 3 次实验。
- 后续实验的配置能够体现对前序实验结果的学习/响应。
- 在 3 次运行中，`val_bpb` 指标整体呈现下降（优化）趋势。

## Out of Scope
- 多节点分布式训练（目前仅限单 GPU）。
- 复杂的图形化监控界面（以 CLI 和日志为主）。
