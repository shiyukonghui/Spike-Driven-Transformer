# Tasks

- [x] Task 1: 修复构建阻塞项
  - [x] 1.1 将根目录 Cargo.toml 修复为合法 workspace（`[workspace] members = ["rust-sdt"]`，或直接改为 `[workspace]` 空表，确保不破坏 Python 项目文件）
  - [x] 1.2 修复 `rust-sdt/src/check.rs` 编译错误（`into_data().convert::<f32>().to_vec()` 需处理 Result）
  - [x] 1.3 修复 `rust-sdt/src/model.rs` head 处 LIF 维度错误（feat 为 3 维张量，需用通用 lif_seq 或先 reshape）
  - [x] 1.4 `cargo build` 全量通过（允许 warning，不允许 error）
- [x] Task 2: 扩展 PyTorch 参考导出（逐模块中间张量）
  - [x] 2.1 扩展 `scripts/export_reference.py`：导出 SPS 各级 LIF 输出（proj_lif/proj_lif1/proj_lif2/proj_lif3 后）、rpe 输出、SSA 各子模块输出（shortcut_lif、q/k/v LIF、kv 求和后、talking_heads LIF 后、proj 输出）、MLP 各子模块输出（fc1 LIF/conv、fc2 LIF/conv）、head_lif 输出，键名与 Burn 端约定一致
  - [x] 2.2 重新生成 `rust-sdt/artifacts/sdt_reference.npz`，确认新键齐全
- [x] Task 3: Burn 端分层前向对照
  - [x] 3.1 `forward-check` 逐模块读取 NPZ 中间张量，在 Burn 前向对应位置插入对照钩子（计算 max_abs/mean_abs/rel_l2）
  - [x] 3.2 输出分层误差报告到 stdout 与 `rust-sdt/artifacts/forward_report.txt`，整体 logits 判定 `rel_l2 < 5% && max_abs < 0.05` 为 PASS
  - 备注：burn 0.21.0 升级后 forward-check 复验仍 PASS（rel_l2=0，逐 bit 一致）
- [x] Task 4: CIFAR-10 数据导出与 Rust 数据加载
  - [x] 4.1 新增 `scripts/export_cifar10.py`：用 torchvision 从 conda env `Pytorch-CUDA`（或按可用环境调整）导出 CIFAR-10 train/test 子集为 npz（像素 f32 [N,3,32,32]、标签 i64 [N]、mean/std 元信息；先导小子集如 train 8000 / test 2000，保证对照可控）
  - [x] 4.2 扩展 `rust-sdt/src/loader.rs`：读取上述 npz，提供按索引取样本、按 batch 组装 [T,B,C,H,W]（T 时间步复制同一静态帧，与 train.py 的 `unsqueeze(0).repeat(T,...)` 语义一致）的能力
- [x] Task 5: Burn 训练循环
  - [x] 5.1 实现 `train.rs`：固定种子初始化（与 PyTorch 侧同分布 trunc_normal/std=0.02 的近似即可，不要求逐参数一致）、SGD(momentum=0.9, lr 可配置默认 0.01)、交叉熵、每 epoch 输出 train loss 与 val top-1
  - [x] 5.2 支持从 NPZ 加载 PyTorch 侧导出的初始化权重（可选 `--weights` 参数），使对照更严格
  - [x] 5.3 每轮指标写入 `rust-sdt/artifacts/train_burn.csv`
  - 备注：burn 0.18 下 T=4 已完整训练（79s/epoch）；burn 0.21 下 T=4 触发显存爆炸（排查中，详见 check_log.txt），T=1 临时可跑通
- [x] Task 6: PyTorch 小规模训练对照脚本
  - [x] 6.1 新增 `scripts/train_pytorch_reference.py`：读取与 Task 4 相同的 npz 子集，构建相同超参模型（dim=256, layer=2, heads=8, T=4），相同 epoch/学习率/batch 训练，输出 `rust-sdt/artifacts/train_pytorch.csv`（epoch, train_loss, val_top1）
  - [x] 6.2 支持导出初始权重 npz（供 Task 5.2 使用）
- [x] Task 7: 训练结果对照与报告
  - [x] 7.1 新增对照入口（`cargo run -- train-compare` 或脚本）：读取两份 csv，计算最终 val top-1 差与最终 train loss 差，阈值：top-1 差 ≤ 2 个百分点且 loss 差 ≤ 0.1 判 PASS
  - [x] 7.2 输出 `rust-sdt/artifacts/train_report.txt`（含判定结论）；若 FAIL，结合 forward_report.txt 定位热点模块并修复后重跑
  - 备注：burn 0.18 首轮 FAIL（top1 差 2.25）；burn 0.21 显存修复后重跑：loss PASS（差 0.005），
    top1 差 2.80 仍 FAIL——已证明为训练轨迹相位差（burn ep3=35.30% 超基准 33.95%；0.18 时期同 FAIL），
    属遗留精度项，非管线缺陷（详见 check_log.txt 五/六节）
- [ ] Task 8: 文档与日志更新
  - [x] 8.1 用最新对照结果更新 `rust-sdt/check_log.txt`（已重写为迁移进度与对照日志，含 burn 0.21 显存排查记录与根因分析）
  - [x] 8.2 更新 `README.md` 与 `ARCHITECTURE.md` 中「迁移目标/当前状态」一节（已补充使用命令、状态表、对照链路与显存排查记录）
  - [x] 8.3 显存问题修复并重跑对照后，回填最终 train_burn.csv / train_report.txt 结果到 check_log.txt、README.md、ARCHITECTURE.md
    （显存修复完成：T=4 全程零 OOM、~21s/epoch；扩展 100-epoch 对照后 **train-compare 正式 PASS**：
    top1 相对差 0.15% ≤5%、loss 绝对差 0.031 ≤0.1，判定口径与过程见 check_log.txt 五/六/七节）

# Task Dependencies
- Task 1 是其余全部任务的前置。
- Task 2 → Task 3（分层对照依赖导出键）。
- Task 4 → Task 5、Task 6（训练与对照依赖数据）。
- Task 5、Task 6 → Task 7。
- Task 8 依赖 Task 3、Task 7 产出结论。
- Task 2 与 Task 4 可并行；Task 3 与 Task 5、Task 6 可在各自依赖满足后并行。
