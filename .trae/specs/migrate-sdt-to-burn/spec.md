# SDT 模型 Rust + Burn 迁移与结果对照 Spec

## Why
原项目为 PyTorch 实现的 Spike-Driven Transformer（SDT）。目标是把模型迁移到 Rust + Burn，并用「前向数值对照 + 小规模训练对照」证明架构迁移误差在允许范围内。当前 `rust-sdt/` 已有前向实现与参考 NPZ，但存在编译错误、训练循环空缺、数据与对照流程缺失，尚不能跑通任何验证。

## What Changes
- 修复根目录无效的 [Cargo.toml](file:///f:/RustProjects/Spike-Driven-Transformer/Cargo.toml)（无 targets 导致 cargo workspace 解析失败，阻塞 `rust-sdt` 全部构建）。
- 修复 `rust-sdt` 全部编译错误（`check.rs` 的 `into_data().to_vec()` 返回 Result、`model.rs` head 处 5 维/4 维 LIF 张量不匹配等），确保 `cargo build` 通过。
- 扩展 [scripts/export_reference.py](file:///f:/RustProjects/Spike-Driven-Transformer/scripts/export_reference.py)：在现有 NPZ 基础上增加「逐模块中间张量」（SPS 各级 LIF 输出、SSA 的 shortcut_lif/q/k/v/kv/talking_heads 输出、MLP 的 fc1/fc2 LIF 输出、`feat_mean`、head_lif 输出），供 Burn 端分层对照。
- 扩展 `rust-sdt` 的 `forward-check`：除整体 logits 对照外，增加逐模块中间张量对照，定位误差来源；输出结构化误差报告。
- 实现 `rust-sdt/src/train.rs`：基于 Burn 的 CIFAR-10 小规模训练循环（数据加载、SGD+momentum、交叉熵、每 epoch 验证、损失曲线输出）。
- 实现 CIFAR-10 数据导出脚本（PyTorch/torchvision 导出 npz → Rust 端读取），或直接下载解析二进制版本（以最稳妥方式为准）。
- 实现 `rust-sdt/src/check.rs` 中的训练对照：固定种子下，Burn 与 PyTorch 各训练相同的小规模轮次（相同数据顺序、相同初始化方式可近似），对照验证精度与损失曲线，误差在允许范围内即视为架构迁移成功。
- 更新 [rust-sdt/check_log.txt](file:///f:/RustProjects/Spike-Driven-Transformer/rust-sdt/check_log.txt) 为最新对照日志（该文件当前记录的是旧编译错误）。
- **不改动** Python 原始训练流程（[train.py](file:///f:/RustProjects/Spike-Driven-Transformer/train.py)、model/、module/、conf/、dvs_utils/），仅作为对照参考保留。

## Impact
- Affected specs: 无既有 spec（本 spec 为首个）。
- Affected code:
  - [Cargo.toml](file:///f:/RustProjects/Spike-Driven-Transformer/Cargo.toml)（根，修复为合法 workspace 空清单）
  - [rust-sdt/src/*](file:///f:/RustProjects/Spike-Driven-Transformer/rust-sdt/src)（check.rs、train.rs、model.rs、loader.rs、main.rs 等）
  - [rust-sdt/Cargo.toml](file:///f:/RustProjects/Spike-Driven-Transformer/rust-sdt/Cargo.toml)（如需补依赖）
  - [scripts/export_reference.py](file:///f:/RustProjects/Spike-Driven-Transformer/scripts/export_reference.py)（扩展中间张量导出）
  - 新增 `scripts/export_cifar10.py`（数据导出）
- Python 参考环境：conda env `Pytorch-CUDA`（torch 2.0.1+cu118、spikingjelly 无 cupy→torch 后端）；timm 在 `.pylibs`。

## ADDED Requirements

### Requirement: Rust 构建可跑通
`rust-sdt` 必须能在当前 Windows 环境下 `cargo build` 与 `cargo run` 成功，无编译错误。

#### Scenario: cargo build 通过
- **WHEN** 在 `rust-sdt/` 目录执行 `cargo build`
- **THEN** 编译成功，无 error（允许 warning）

### Requirement: 前向数值对照（整体 + 分层）
Burn 前向输出必须与 PyTorch 参考输出做数值对照，且提供分层中间张量对照以定位误差。

#### Scenario: 整体 logits 对照通过
- **WHEN** 执行 `cargo run -- forward-check`
- **THEN** Burn logits 与 PyTorch logits 的 `rel_l2 < 5%` 且 `max_abs < 0.05`，输出 `PASS`

#### Scenario: 分层对照定位误差
- **WHEN** 执行带分层对照的 forward-check
- **THEN** 依次输出 SPS、每个 block 的 SSA/MLP 中间张量的误差统计（max_abs / mean_abs / rel_l2），可据此定位首个超阈值模块

### Requirement: Burn 训练循环可用
`cargo run -- train` 必须在 CIFAR-10 上完成小规模训练：数据加载、SGD(momentum=0.9)、交叉熵损失、每 epoch 结束输出 train loss 与 val top-1。

#### Scenario: 训练若干 epoch
- **WHEN** 执行 `cargo run -- train --epochs 2`
- **THEN** 输出每 epoch 的 loss 与 val top-1，且 val top-1 显著高于随机水平（10%），证明前向+反向+优化链路正确

### Requirement: 训练结果对照（架构迁移误差验证）
固定随机种子，Burn 与 PyTorch 使用相同的模型超参（dim=256, layer=2, heads=8, T=4, 32x32）、相同的数据子集与相同的训练轮数分别训练，对照两者最终验证精度差。

#### Scenario: 训练对照通过
- **WHEN** 双方各自完成相同小规模训练（如 2-3 epoch、限定样本数）
- **THEN** 两者 val top-1 之差 ≤ 2 个百分点，且两者损失曲线趋势一致（最终 train loss 差 ≤ 0.1），判定架构迁移误差在允许范围内，输出对照报告

#### Scenario: 训练对照不通过
- **WHEN** 上述差距超阈值
- **THEN** 输出 `FAIL` 并给出分层前向对照的误差热点，指导修复后重跑

### Requirement: 对照产物留存
对照过程的关键数字必须落盘，便于复盘。

#### Scenario: 产物文件生成
- **WHEN** forward-check 与 train 对照执行完毕
- **THEN** `rust-sdt/artifacts/` 下生成/更新对照报告（如 `forward_report.txt`、`train_report.txt`），包含全部误差统计与判定结论

## MODIFIED Requirements
（无——本项目此前没有 spec 化的需求基线）

## REMOVED Requirements
（无）
