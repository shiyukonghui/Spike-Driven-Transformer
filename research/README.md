# SDT 脉冲注意力的 ES 训练与 QAM 加速——研究总结（2026-09-08 ~ 09-09）

本文件夹是全部研究过程与结论的**汇总快照**。权威的过程日志（持续追加式）
在 [`rust-sdt/check_log.txt`](../rust-sdt/check_log.txt)（十五节）与
[`rust-sdt/ES_MANIFOLD_NOTE.md`](../rust-sdt/ES_MANIFOLD_NOTE.md)（13 节）；
本目录将其提炼为可独立阅读的四个文档。

## 研究问题

1. 无梯度方法（Evolution Strategies, ES）能否有效训练脉冲 Transformer（SDT）？
2. 从 QAM-SNN（正交幅度调制，H:\snn-thinking\agid\qam-snn）引入可学习调制，
   能否加速 SDT 的 ES 训练？（指定路线：数学 → 软件 → 大规模训练）
3. ES 的价值边界在哪里（混合估计器、数据规模、与 SGD 的关系）？

## 路线图

```
数学框架（流形分析/N-D律/TSES/QAM对偶/位点深度律）
  → Python 数值验证（verify_qam_math.py c1/c2/c3 全过）
  → rust-sdt 软件实现（QAM 参数槽，commit 65fb23d）
  → 10ep A/B → 1000ep Run-3（子集）→ Run-4 σ衰减（证伪并回退）
  → 混合估计器实现+三臂对照（净负结果，价值边界划定）
  → Run-5 全量 CIFAR-10（ES 线新纪录 18.24%）
```

## 结论速览

| 路线 | 最优配置 | best_val | 定位 |
|---|---|---|---|
| 纯 ES（子集 8000） | factored + TSES + lr 调度 | 17.30%（Run-2，1000ep） | 无梯度基线 |
| 纯 ES + QAM ssa（子集） | 同上 + `--qam learnable --qam-sites ssa` | 17.15%（Run-3，但平台提前 ~2×） | 加速不加天花板 |
| 纯 ES + QAM ssa（**全量 50000**） | Run-3 配方 × 全量 | **18.24%**（Run-5，60ep） | ES 线纪录 |
| 纯 SGD（子集） | momentum 0.9 | 66.10%（100ep） | 梯度可用时的正确基线 |
| 纯 SGD（全量） | 同上 | 58.95%（仅 5ep） | 数据杠杆质变 |
| 混合（SGD + ES 低维） | v_th + QAM | 58.70%（劣于纯 SGD 7pp） | **净负**，见 04 文档 |

**一句话结论**：ES/QAM 路线的价值域 = 完全无梯度约束的场景（硬量化部署、
黑盒仿真器、不可反传硬件）；QAM 是其中的效率优化（等精度省 2–3× 墙钟），
不抬高渐近精度；梯度可用时 SGD 全权管理是最优解。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01_数学框架.md](01_数学框架.md) | 平板流形引理、方差级联与 N/D 律、温度松弛定理、QAM 增益-阈值对偶、位点深度律、T 压缩假设 |
| [02_实现与验证.md](02_实现与验证.md) | rust-sdt 软件架构要点、QAM/混合估计器实现细节、数值验证方法与结果 |
| [03_实验全记录.md](03_实验全记录.md) | v1→v4、Run-1→Run-5、混合估计器三臂、SGD 基线——全部配置与结果 |
| [04_结论与价值边界.md](04_结论与价值边界.md) | 六条主结论、被证伪假设清单、三路线价值边界、后续建议 |
| [05_ES协议重审_多角度分析.md](05_ES协议重审_多角度分析.md) | **2026-09-09**：以原始库代码为准重审——协议合规 diff、六个新角度（评估契约/适应度整形/solver/逐层工作点/σ 参考系/训练时长）、结论修正声明、决定性实验矩阵 |
| [06_论文检索与思路迁移.md](06_论文检索与思路迁移.md) | **2026-09-09~10**：ModelScope 检索 API（paper-cli）+ X1-X7 系列实验（锚点/σ自适应/普通SNN/MNIST 复现/robust-z/hard/泊松/秩64） |
| [07_ES阶段总结.md](07_ES阶段总结.md) | **2026-09-10 阶段收官**：两线最终位置（SDT 23.96 / MNIST 87.34）、已证规律、用户"参数探索无根本突破"判断的确认与根本性方向候选 |

## 复现索引（关键命令）

```bash
# ES 线纪录（Run-5，全量 + QAM ssa）
cargo run --release -p rust-sdt -- train-es --epochs 60 --mode factored \
  --pop 50000 --es-batch 1 --chunk 256 --data-dir ../data/cifar10_full \
  --weights artifacts/train_burn_init.npz --no-calibrate \
  --sigma 0.5 --rank 32 --lr 0.02 --relax --beta 4.0 \
  --beta-anneal-every 25 --relax-scope all --validate-every 2 \
  --lr-warmup 25 --lr-min-frac 0.1 \
  --qam learnable --qam-sites ssa \
  --csv-out artifacts/train_burn_es_full_qam.csv

# 混合估计器（负结果对照）
cargo run --release -p rust-sdt -- train-mixed --epochs 100 \
  --weights artifacts/train_burn_init.npz --no-calibrate \
  --qam learnable --qam-sites ssa --csv-out artifacts/train_mixed100.csv

# 数学验证
python scripts/verify_qam_math.py
```

全部 CSV/日志在 `rust-sdt/artifacts/`；关键 commit：
d35a972（QAM 数学）→ 65fb23d（QAM 实现）→ e186a78（混合估计器）→ 1e6ecc1（Run-5）。
