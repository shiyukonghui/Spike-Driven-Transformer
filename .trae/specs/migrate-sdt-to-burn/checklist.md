# Checklist

- [ ] 根目录 Cargo.toml 为合法 workspace 清单，`cargo build`（在 rust-sdt/ 下）无 error
- [ ] `cargo run -- forward-check` 整体 logits 对照：rel_l2 < 5% 且 max_abs < 0.05，输出 PASS
- [ ] forward-check 分层对照覆盖 SPS 全部 LIF 级、每个 block 的 SSA（shortcut/q/k/v/kv/talking_heads/proj）与 MLP（fc1/fc2），误差统计齐全
- [ ] `rust-sdt/artifacts/forward_report.txt` 已生成，含全部误差统计与判定结论
- [ ] `rust-sdt/artifacts/cifar10_data.npz`（或同名数据 npz）已由导出脚本生成，Rust 端可加载并按 batch 组装 [T,B,C,H,W]
- [ ] `cargo run -- train` 可完成小规模训练，每 epoch 输出 train loss 与 val top-1，最终 val top-1 明显高于 10% 随机水平
- [ ] `rust-sdt/artifacts/train_burn.csv` 与 `rust-sdt/artifacts/train_pytorch.csv` 均已生成，训练超参与数据子集一致
- [ ] train-compare 对照判定：最终 val top-1 差 ≤ 2 个百分点，且最终 train loss 差 ≤ 0.1，输出 PASS
- [ ] `rust-sdt/artifacts/train_report.txt` 已生成，含判定结论；若曾 FAIL，有热点定位与修复记录
- [ ] `rust-sdt/check_log.txt` 已更新为最新对照结果（不再是旧编译错误日志）
- [ ] README.md 与 ARCHITECTURE.md 的迁移状态/使用说明已更新
- [ ] 未改动 Python 原始训练流程（train.py、model/、module/、conf/、dvs_utils/）的既有语义
