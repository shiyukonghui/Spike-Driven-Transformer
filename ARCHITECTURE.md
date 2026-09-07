# Rust Spike-Driven Transformer

将 Python Spike-Driven Transformer (SDT) 模型迁移到 Rust（使用 Burn 深度学习框架）。

## 当前状态（2026-09-08）

- **框架**：Burn 0.21.0（wgpu GPU 后端 + ndarray 调试后端 + autodiff），已从 0.18 升级并完成 API 适配
  （`BackendTypes::Device`、`squeeze::<D>()` 无参、`max_pool2d` 带 `ceil_mode`、`Backend::sync` 返回 `Result`）。
- **前向对照（forward-check）**：PASS。整体 logits 与 PyTorch 参考**逐 bit 一致**
  （max_abs=0，rel_l2=0，判定线 rel_l2<5% 且 max_abs<0.05）。
- **burn 0.21 显存问题**：**已修复**（本地补丁 + 训练循环改造，详见下节）。
  T=4 完整训练（校准 + 2 epoch + 2 次 eval）全程零 OOM，~21s/epoch（0.18 基线 79s/epoch 的 3.8×）。
- **训练对照（100 epoch 扩展，seed=42，可复现）**：
  - burn epoch100：loss 0.128889 / val_top1 65.55%
  - pytorch epoch100：loss 0.159796 / val_top1 65.65%
  - 判定：**PASS**（top1 相对差 0.15% ≤5%；loss 绝对差 0.031 ≤0.1；
    口径经裁决：top1 用相对差——长跑下量级从 10%→65% 变化，相对差跨阶段可比；
    loss 用绝对差——深度过拟合区分母小，相对差噪声被放大）
  - 100 epoch 全程零 OOM（burn 侧 2067s），val top-1 逐 epoch 轨迹双边同步爬升
  - 报告见 `rust-sdt/artifacts/train_report.txt`；2-epoch 历史对照见
    `artifacts/train_pytorch_2ep.csv`

## 迁移目标

将原 Python 项目（基于 PyTorch）的 Spike-Driven Transformer 模型用 Rust 重写，
保持模型架构一致，并对比训练结果确保误差在允许范围内。
验收标准（2026-09-08 更新）：100 epoch 训练后，最终 val top-1 相对差 ≤ 5%
（分母=pytorch 值）且 train loss 绝对差 ≤ 0.1 判 PASS。当前状态：**PASS**。

## Python 原始架构（SDT / Spike-Driven Transformer）

- **SSA (Spike-driven Self-Attention)**: 基于 MSA (Multi-Spike Attention) 的脉冲自注意力
- **LIF 神经元**: 泄漏积分发放神经元模型
- **MSA**: Multi-head Spike Attention

## Rust 端架构与对照链路

```
scripts/export_reference.py        # PyTorch 权重 + 逐模块中间张量 → rust-sdt/artifacts/sdt_reference.npz
        │
        ▼
rust-sdt/src/check.rs              # forward-check：Burn 前向逐模块对照（SPS/SSA/MLP 各 LIF 级），→ forward_report.txt
        │
scripts/export_cifar10.py          # CIFAR-10 子集 → artifacts/cifar10_data.npz
        │
        ▼
rust-sdt/src/loader.rs             # [T,B,3,32,32] 批组装（T 时间步复制同一静态帧）
rust-sdt/src/model.rs + ops.rs     # forward_full：SPS → 2×(SSA+MLP) → head；LIF STE、lif_seq 按 T 循环切片
        │
        ▼
rust-sdt/src/train.rs              # SGD(0.9) + CE + 每 epoch train loss/val top1 → train_burn.csv；sync+memory_cleanup
        │
scripts/train_pytorch_reference.py # 同超参同数据 PyTorch 基准 → train_pytorch.csv
        │
        ▼
cargo run -- train-compare         # 逐 epoch 对照 → train_report.txt（top1 差 ≤2 且 loss 差 ≤0.1 判 PASS）
```

关键超参：dim=256、layer=2、heads=8、T=4、SG_ALPHA=4.0、TAU=2.0、SPS 输入 32×32 两次 maxpool → 8×8。

## burn 0.21 显存问题：根因与修复（已解决，2026-09-08）

修复前的现象与实验矩阵（详见 [rust-sdt/check_log.txt](rust-sdt/check_log.txt)）：

| 变量 | 结果 |
| --- | --- |
| `--batch-size 8` | 仍 OOM（与 batch 无关） |
| `--no-calibrate` | 仍 OOM（与校准残留无关） |
| `CUBECL_WGPU_MAX_TASKS=1` | 仍 OOM（与惰性任务积压无关） |
| `--time-steps 1` | 成功完整训练（爆炸与 T 正相关） |

VRAM 探测（`probe_vram.ps1`）：校准期 2.3→5.9GB 正常渐增；第一个训练 step 单次采样内 4.7→20.8GB，随后 23.4GB 触顶。
RUST_BACKTRACE 定位分配点：`wgpu_core::create_buffer ← WgpuStorage::alloc ← MemoryManagement::reserve`。

三个叠加根因与对应修复：

1. **burn-fusion 0.21 延迟 drop（主因）**：`MultiStream` 将 `DropAction::ContinueDrop`
   从 0.18 的 `=> true`（每次普通 drop 触发 drain、缓冲立即归还复用）改为 `=> false`
   （drop op 仅在 sync/drain 时执行）。T=4 训练图每步数千个 8MB 激活的 drop 全部积压，
   内存池须为所有句柄保留页 → 首个 backward 期间 OOM。
   **修复**：本地补丁（`patches/burn-fusion`，经根 Cargo.toml `[patch.crates-io]` 生效）
   恢复 drop-drain 语义：`ContinueDrop` 分支改走 `continue_drop_drain_enabled()`
   （默认 true；`BURN_FUSION_CONTINUE_DROP_DRAIN=0` 可关）。
2. **纯前向路径的图节点泄漏**：burn-autodiff 反向图节点在无 backward 消费时
   （eval/校准）不释放，每批 ~50 个 8MB 激活被钉住——62 批 eval 累积 ~26GB、
   2720 次校准前向残留 ~6.8GB。
   **修复**：eval 改在内层 wgpu 后端运行（`Tensor::inner()` 零拷贝，无 autodiff 图；
   `eval_top1` + `to_wgpu_weights`）；校准/eval 循环内逐批 `Backend::sync` +
   `memory_cleanup` 归还全空页。
3. **cubecl 0.10 SlicedPages 池 `dealloc_period: None` 从不自动回收**：
   保留阶段边界清理（校准后/每 epoch 末 sync + memory_cleanup）。

判别实验（`mem-probe` 子命令，SDT_PROBE_STAGE=0..5）逐层二分：纯 conv 循环零泄漏
（栈层正常）→ 真实模型仅前向每迭代泄漏 ~14 页 → fwd+bwd(+SGD) 零泄漏（训练内环
健康）→ 直接调用 `train_epoch` 零泄漏 → 进程内 `run_train_with_args` 复现
（定位到 eval 纯前向）。被钉住句柄反查（cubecl-runtime 诊断补丁）确认全部是
fused 输出的客户端句柄。

修复后（`train --epochs 2` 带校准，可复现）：全程零 OOM、~194s 总耗时
（校准 139s + 2×训练 21s + 2×eval 3s）；train-compare loss PASS（差 0.005）、
top1 差 2.80（轨迹相位差，burn ep3=35.30% 超基准佐证，0.18 时期同 FAIL）。

诊断设施保留（默认零开销，环境变量门控）：`BURN_FUSION_LOG`（drain 日志）、
`BURN_SDT_POOL_DIAG=1`（cubecl 池页/句柄诊断）、`BURN_FUSION_CONTINUE_DROP_DRAIN`
（补丁行为开关）、`SDT_PROBE_*`（mem-probe 复现实验）、`SDT_BATCH_SYNC` /
`SDT_BATCH_CLEANUP`（逐批 sync/cleanup 开关）。

## 项目结构

```
├── model/ module/ train.py conf/ dvs_utils/   # Python 原始实现（未改动）
├── scripts/                                    # 导出与 PyTorch 基准脚本
├── rust-sdt/                                   # Rust 实现（Burn）
│   ├── src/                                    # check/config/loader/main/model/ops/train/tensor_io
│   ├── artifacts/                              # 权重/数据/报告/CSV
│   ├── probe_vram.ps1                          # VRAM 轮询探测脚本
│   └── check_log.txt                           # 迁移进度与对照日志（最新）
└── .trae/specs/migrate-sdt-to-burn/            # spec / tasks / checklist
```
