# Saved Benchmark Findings

这份文件只保留后续最值得复用的结论，适合在重新开始性能优化前快速回顾。

## 1. 当前默认基线 (Updated 2026-03-16)

- 当前默认 `baseline/default` 基线是在原 `mfu50` 路线上加了 MLP-only checkpoint 后，把模型加深到 `9L/768d` 的版本。
- **2026-03-16 优化**: 引入了 `WARMUP_RATIO = 0.05` 和 `MUON_WARMUP_STEPS = 100`。
- 代表记录：`generated/2026-03-16/opt_val_bpb_20260316.txt` (推算)
- 配置：
  - `n_embd=768`
  - `depth=9`
  - `WARMUP_RATIO=0.05`
  - `MUON_WARMUP_STEPS=100`
- 参考结果：
  - `val_bpb: 2.526` (新记录)
  - `25,320 tok/s`
  - `43.94% MFU`

## 2. 吞吐对照基线

- 原来的 throughput-first 基线现在保留为 `throughput` profile。
- 代表记录：`generated/2026-03-15/baseline_default_20_serial.txt`
- 配置：
  - `n_embd=512`
  - `depth=8`
  - `MAX_SEQ_LEN=2048`
  - `WINDOW_PATTERN=SSSL`
  - `DEVICE_BATCH_SIZE=12`
  - `compile_backend=inductor`
  - `compile_mode=default`
  - `compile_scope=model`
  - `optimizer_compile_backend=inductor`
- 参考结果：
  - `73,537 tok/s`
  - `17.58 train TFLOPS`
  - `40.16% MFU`
  - `4610.3 MB peak VRAM`

## 3. 已经确定的方向判断

- `max-autotune` 在当前 Windows + `triton-windows` 路径下不适合作默认值。
- `max-autotune-no-cudagraphs` 比 `max-autotune` 更实用，但整体仍不如 `default`。
- 提高 MFU 的有效杠杆不是单纯增大 batch，而是：
  - 保持 `768d` 左右的宽度，并优先用更深层数去放大模型
  - 使用 `LLLL`
  - 拉长 `MAX_SEQ_LEN`
  - 同时把 batch 控制在不会掉进显存退化区的范围内
- MLP-only checkpoint 是目前最值得保留的 activation checkpoint 形式。
  它比“整层 checkpoint”更均衡，足够给默认基线腾出显存去容纳更宽的模型。
- `9L/768d` 是当前机器上更深默认值的合理上限。
  `10L/768d` 已经降到约 `46.6% MFU`，再把 batch 拉到 `6` 会直接掉进显存退化区。
- 旧版 `896d`, `10L`, `batch=8` 的 `mfu50` 路线已经证伪，不建议继续投入。

## 4. 读旧日志时要注意

- `3` 步和 `5` 步日志主要用于筛方向，不是最终性能排名。
- `baseline_default_20.txt` 是并发占卡导致的无效样本，不要引用。
- `modular_baseline_3_after_warmup.txt` 和 `modular_mfu50_3.txt` 的价值是“重构后性能未退化”，不是替代 20 步或 100 步主 benchmark。

## 5. 详细记录入口

- 详细表格版总结见 `SUMMARY_2026-03-15.md`
- 历史归档日志见 `archive/`
- 本地生成日志见 `generated/`
