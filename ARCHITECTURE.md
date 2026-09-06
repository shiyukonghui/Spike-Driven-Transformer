# Rust Spike-Driven Transformer

将 Python Spike-Driven Transformer (SDT) 模型迁移到 Rust（使用 Burn 深度学习框架）。

## 当前状态

Rust 实现迁移进行中。由于 Windows 环境下的依赖编译问题，完整的 cargo 构建被推迟。

## 迁移目标

将原 Python 项目（基于 PyTorch）的 Spike-Driven Transformer 模型用 Rust 重写，
保持模型架构一致，并对比训练结果确保误差在允许范围内。

## Python 原始架构（SDT / Spike-Driven Transformer）

- **SSA (Spike-driven Self-Attention)**: 基于 MSA (Multi-Spike Attention) 的脉冲自注意力
- **LIF 神经元**: 泄漏积分发放神经元模型
- **MSA**: Multi-head Spike Attention

## 环境准备（已完成）

- PyTorch 2.0.1 (CPU) + Python 3.11 环境
- Spike-Driven Transformer 完整实现
- 支持 CIFAR-10 等数据集训练

## 项目结构

```
├── models/           # Python 模型定义
├── rust-spike/       # Rust 实现 (待创建)
├── sdt/              # Spike-driven Transformer 核心模块
└── ...
```
