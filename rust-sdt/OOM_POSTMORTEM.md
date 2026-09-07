# burn 0.21 T=4 训练显存爆炸：发现、复现、根因、修复与验证

> Postmortem 日期：2026-09-08
> 环境：Rust workspace（rust-sdt），Burn 0.21.0（burn-cubecl + burn-cubecl-fusion，wgpu 后端），cubecl 0.10.0，RTX 4090 24GB，Windows，驱动 616.56
> 关联文档：[check_log.txt](check_log.txt)（进度日志）、[../ARCHITECTURE.md](../ARCHITECTURE.md)（架构与状态）

---

## 0. TL;DR

burn 0.18 → 0.21 升级后，T=4（默认时间步）训练在**第一个训练 step** 触发 1.4 万次
wgpu Out of Memory 并连锁 panic，24GB 显存触顶。排查确认三个叠加根因：

1. **burn-fusion 0.21 延迟 drop**（主因）：普通张量 drop 不再触发 drain（0.18 会触发），
   激活释放全部积压 → 本地补丁恢复 0.18 drop-drain 语义；
2. **纯前向路径（eval/校准）autodiff 图节点不释放**：每批全部激活被钉住，
   62 批 eval 累积 ~26GB → eval 改走内层 wgpu 后端 + 逐批 sync/cleanup；
3. **cubecl SlicedPages 池 `dealloc_period=None` 从不自动回收** → 保留阶段边界清理。

修复后 `train --epochs 2`（带校准）全程**零 OOM**，~21s/epoch（0.18 基线 79s/epoch 的
**3.8× 提速**），结果可复现；train-compare loss PASS（差 0.005），top1 差 2.80 为
训练轨迹相位差的离散放大（burn ep3=35.30% 超基准 33.95% 佐证，非管线缺陷）。

---

## 1. 背景

- SDT（Spike-Driven Transformer）已从 PyTorch 迁移到 Rust/Burn，前向对照
  （forward-check）与 PyTorch **逐 bit 一致**。
- burn 0.18 时期 T=4 训练可完整跑通（79s/epoch），train-compare：loss PASS、
  top1 差 2.25 → FAIL（当时遗留）。
- 升级 burn 0.21.0（wgpu + ndarray + autodiff）后，T=4 训练无法开始。

## 2. 现象

- 校准阶段（纯前向，~2720 次前向）显存 2.3→5.9GB 渐增；校准后 sync+cleanup 仍残留
  4.7~6.8GB（当时未解释，后归因于根因二）。
- **第一个训练 step**：单次 VRAM 采样内 4.7→20.8GB（+16GB），随后 23.4GB 触顶。
- `wgpu error: Out of Memory` **13832 次**，随后连锁
  `Should have handle for tensor` panic（burn-ir `handle.rs:97`）与
  `ordering.rs:65 index out of bounds`，进程 exit=101。
- `RUST_BACKTRACE` 定位分配点（真实显存分配失败，非句柄错误）：

  ```
  wgpu_core::create_buffer
    ← WgpuStorage::alloc
    ← MemoryManagement::reserve
    ← WgpuServer::initialize_memory
  ```

- 单张量并不大（最大 fc1 输出 [T*B,1024,8,8] ≈32MB），单步 +16GB 指向
  "激活数 × 页级分配"的复合放大，与 T（时间步数，放大激活数）正相关的观测吻合。

## 3. 第一阶段：静态判别实验（变量矩阵）

| 变量 | 结果 | 结论 |
| --- | --- | --- |
| `--batch-size 8` | 仍 OOM | 与 batch 无关 |
| `--no-calibrate` | 仍 OOM | 与校准残留无关（但残留是独立问题，见 §7） |
| `CUBECL_WGPU_MAX_TASKS=1`（后验证 =4） | 数值完全不变 | 与惰性任务积压/绑定反压无关 |
| `CUBECL_AUTOTUNE_LEVEL=0` | 无变化 | 与 autotune 无关 |
| `--time-steps 1` | 完整跑通（15.4s/epoch，top1=10.55%≈随机） | **爆炸与 T 正相关**（T 放大激活数量） |
| `SDT_BATCH_CLEANUP=1` | 不解决 | cleanup 只还"全空页"，钉住页还不掉 |

同时完成启动路径审计：权重加载（NPZ→wgpu）、`to_autodiff_weights`、模型构造、
SGD 初始化均无泄漏。静态读码排除了：`HandleContainer`（存活句柄有界 ~300，而泄漏
数百页）、`PersistentPool`、trace/OutputPlanner（生命周期短）、launch 绑定路径
（binding 被 consume，不持 descriptor）、pipeline 缓存（只存编译产物）、任务队列
（task 执行后即弃）。

## 4. 第二阶段：根因一 —— burn-fusion 0.21 延迟 drop（主因）

**差异点**（`patches/burn-fusion/src/stream/multi.rs`）：

```rust
// 0.18：每次普通张量 drop（ContinueDrop）都触发 drain → 缓冲立即归还复用
Some(DropAction::ContinueDrop) => true,
// 0.21：drop op 仅在 sync / drain 时执行 → 激活释放全部积压
Some(DropAction::ContinueDrop) => false,
```

T=4 训练图每步产生数千个 8MB 级激活；drop 积压期间内存池必须为所有"已死但未释放"
的句柄保留页 → 首个 backward 的分配请求全部落在新页 → OOM。

**修复（本地补丁，经根 Cargo.toml `[patch.crates-io]` 生效）**：

```rust
// patches/burn-fusion/src/stream/multi.rs
Some(DropAction::ContinueDrop) => continue_drop_drain_enabled(),  // 默认 true
```

环境变量 `BURN_FUSION_CONTINUE_DROP_DRAIN=0` 可关闭（回归对照用）。
打补丁后训练循环完整跑通（~21s/epoch），但显存仍随批次线性增长（~60MB/批）——
进入第三阶段。

## 5. 第三阶段：钉住者追踪（"谁多持了一份引用"）

补丁后残留现象：drop 执行了（补丁侧 drop_8mb 计数器数千次），但 8MB 池页数仍线性
增长（955 页 / 250 批），free 切片恒 ~25，强引用计数直方图中 **c2（count=2，
页 + 一个外部克隆）线性上涨** → 每个泄漏张量恰好多一份外部引用。

**cubecl-runtime 诊断补丁**（`BURN_SDT_POOL_DIAG=1` 启用，默认零开销）：

- `MemoryPage::strong_count_histogram()` / `pinned_ids()`：每池每 100 新页打印
  页数/切片数/引用计数直方图；
- `alloc_sizes` 注册表：`reserve` 记录 (分配 id → 请求大小)，`bind` 时为
  `client.empty()` 客户端句柄补记真实大小（其大小 = reserved 切片请求大小）；
- 诊断打印时**反查被钉住切片的来源大小**与分配序号。

**反查结果（决定性数据）**：

```
8MB 池  pinned_sizes=[(8388608, 890), (32768, 560), ...]     → 890 × 8MB  ≈ 7.4GB
33MB 池 pinned_sizes=[(2097152, 2186), (4194304, 200), (33554432, 99), ...] ≈ 8.7GB
pinned_ids=[0,1,2,...68, 346469, 346470, ...]
```

- 序号 0~68 = 模型权重（**正常**的永久钉住）；
- 346K+ = 持续新增的**死张量**，且**全部是 fused 输出的客户端句柄**
  （`client.empty()` 创建，不经 `reserve`——反查表为空即证明）；
- 合计 ~13GB 死显存，与池增长速率吻合。

逐一排除嫌疑后仍无法从静态代码定位"那一份克隆"的持有者（burn-cubecl-fusion
~16K 行），转入**运行时二分**。

## 6. 第四阶段：mem-probe 分层二分（决定性）

新增 `rust-sdt` 子命令 `mem-probe`（`SDT_PROBE_STAGE=0..5` 逐级逼近真实训练），
配合池诊断观察每级是否新增池页：

| stage | 内容 | 8MB 池增长 | 结论 |
| --- | --- | --- | --- |
| 0 | 纯 conv [128,256,8,8] fwd+bwd 循环 ×300 | **0 页** | 栈层（cubecl/burn 通用路径）复用正常 |
| 1 | 真实模型**仅前向**+标量读 ×100 | ~650 页 | **纯前向每迭代泄漏 ~14 页** |
| 2 | +backward ×100 | 0 页 | backward 消费图 → 释放正常 |
| 3 | +SGD step（含 NPZ 权重/真实数据/逐批 sync）×600 | **0 页** | **训练内环完全健康** |
| 4 | 直接调用 `train_epoch` 本体 ×80（NPZ 权重+真实数据+前置 sync/cleanup/权重存活等逐一对照） | **0 页** | 排除训练循环与其全部输入差异 |
| 5 | 进程内调用 `run_train_with_args` | 复现（eval 阶段 OOM） | 泄漏在**入口包装的 eval（纯前向）** |

结论：**fwd+bwd 不泄漏，纯前向泄漏**。backward 会显式消费图节点并逐级 drop 激活；
无 backward 时图节点（连同其 GPU 缓冲句柄）不释放。这与 burn 上游已知问题同源
（[#2487 GraphMemoryManagement 节点不释放](https://github.com/tracel-ai/burn/issues/2487)、
[fix #2347](https://github.com/tracel-ai/burn/commit/0592af0e8a293b6a4fc3106873ec63c77cd0d692)）。

## 7. 根因二 —— 纯前向路径泄漏与修复

- **eval**：62 批 × ~400MB（~50 个 8MB 激活/批）≈ **26GB** → 这就是"训练 250 批
  跑通后、eval 第 20~30 批崩"的真因（此前误判为训练期泄漏：池时间线混入了 eval 页）。
- **校准**：2720 次纯前向累积 ~6.8GB 残留（修复前校准后残留 4.7~6.8GB 的来源）。

**修复**：

1. `eval_top1` 改在**内层 wgpu 后端**运行：`to_wgpu_weights()` 用
   `Tensor::inner()` 零拷贝包装（无数据拷贝、无 autodiff 图），中间张量走普通
   drop 路径，复用正常；
2. 校准（`layer_output_stats`）与 eval 循环内**逐批 `Backend::sync` +
   `memory_cleanup`**（顺序约束：先 sync 后 cleanup，否则与 fusion 后台线程
   并发访问句柄容器触发竞态 panic——实测复现过）。

## 8. 根因三 —— cubecl 池不自动回收

cubecl 0.10 默认 SubSlices 策略：`SlicedPages` 池 `dealloc_period: None` 从不自动
回收；`max_page_size = heap/4`（24GB 卡 → 6GB 大页）；`memory_cleanup` 仅显式调用
时释放全空页。保留阶段边界清理（校准后/每 epoch 末 sync + cleanup），并与逐批
清理配合覆盖全部死张量归还路径。

## 9. 验证

### 9.1 显存行为（核心目标）

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| `train --epochs 2`（带校准） | 首个 step OOM 13832 次，exit=101 | **全程零 OOM，exit=0，总 194s** |
| eval（62 批 × 2 epoch） | eval 20~30 批内 OOM 140~150 次 | 全程零 OOM（每批 ~2.8s） |
| `--no-calibrate` 全程 | OOM | 50.3s 跑通，零 OOM（loss 停在 ln(10)≈2.30、top1≈10% 为未校准 LIF 死锁的预期行为） |
| 复现性 | — | 两次完整运行 loss/top1 **逐位一致**（2.075424/31.10、1.825703/31.15） |

### 9.2 性能

- epoch 训练 ~21s（0.18 基线 79s/epoch 的 **3.8×**）；
- 校准 139s（17 层 × 10 轮 × 16 批逐层统计，含逐批清理开销）。

### 9.3 数值对照（train-compare，seed=42 两侧一致）

| 指标 | burn (ep2) | pytorch (ep2) | 差 | 判定 |
| --- | --- | --- | --- | --- |
| train_loss | 1.825703 | 1.830727 | 0.005024 | **PASS**（≤0.1） |
| val_top1 | 31.15% | 33.95% | 2.80 | FAIL（≤2.0，遗留项） |

**top1 差距归因**（证据链）：
- train loss 双边同步下降（差 0.005）→ 训练动态几乎一致；
- burn 3-epoch 诊断跑：ep1 31.10 → ep2 31.15（平台）→ **ep3 35.30% 超过基准 33.95%**
  → 管线能力无缺失；
- 0.18 时期同判据同样 FAIL（diff 2.25）→ 非本次修复引入的回归；
- 定性：同 seed 下跨框架 backward 数值差异（cuDNN vs cubecl kernel）造成轨迹相位差，
  在 2000 样本 val top-1 的离散统计上放大。收敛路径（需与基准语义协商）：
  扩展基准到 3 epoch 做 ep3-vs-ep3 对照，或多 seed 评估 2-epoch 窗口的 top1 方差。

**100-epoch 扩展验证（2026-09-08，最终判定 PASS）**：双侧扩展到 100 epoch
（pytorch: 1753s；burn: 2067s 全程零 OOM）。ep100 终点：burn loss 0.128889 /
top1 65.55% vs pytorch loss 0.159796 / top1 65.65%。判定（口径经裁决）：
val_top1 相对差 0.15% ≤5% **PASS**；train_loss 绝对差 0.031 ≤0.1 **PASS**——
2-epoch 窗口的 top1 差确认为轨迹相位噪声，深度长跑后两边泛化行为收敛一致。

## 10. 保留的诊断设施（默认零开销，环境变量门控）

| 设施 | 位置 | 开关 |
| --- | --- | --- |
| ContinueDrop→drain 修复 + drain 频率日志 | `patches/burn-fusion/src/stream/multi.rs`、`tensor.rs` | `BURN_FUSION_CONTINUE_DROP_DRAIN`（默认开）、`BURN_FUSION_LOG` |
| 池页/句柄诊断（每 100 新页：页数/切片/引用直方图/钉住反查） | `patches/cubecl-runtime/src/memory_management/*`、`src/alloc_sizes.rs` | `BURN_SDT_POOL_DIAG=1` |
| mem-probe 分层复现 | `rust-sdt/src/train.rs` `run_mem_probe` | 子命令 `mem-probe` + `SDT_PROBE_STAGE/REALDATA/NPZWEIGHTS/SYNC/PRECLEAN/KEEPWGPU` |
| 逐批 sync / cleanup | `rust-sdt/src/train.rs` `train_epoch` | `SDT_BATCH_SYNC`（默认开）/`SDT_BATCH_CLEANUP`（默认关） |
| VRAM 轮询探针 | `rust-sdt/probe_vram.ps1` | `powershell -File probe_vram.ps1`（注意用 `powershell` 而非 `pwsh`） |
| 批进度打印（每 10 批，对照池时间线） | `train_epoch` / `eval_top1` | 常开 |

## 11. 经验教训

1. **先建最小复现，再读代码**：burn-cubecl-fusion ~16K 行的静态排查消耗了大量
   轮次；`mem-probe` 半天锁定到行级机制（前向 vs 前向+backward）。
2. **分层二分最有效**：栈层（通用 op）→ 真实模型内环 → 入口包装，每层一个开关，
   逐级逼近；"直接调用被怀疑函数本体"（stage 4 调 `train_epoch`）是消除实现
   差异歧义的杀手锏。
3. **"谁被钉住"比"谁没释放"更好答**：分配 id→大小注册表 + 引用计数直方图反查，
   一次把 13GB 死显存映射到具体张量类别（fused 输出客户端句柄），把嫌疑面从
   全栈缩到一个创建路径（`client.empty()`）。
4. **池时间线要按阶段切分**：训练期与 eval 期的池增长曾混在一张时间线里，导致
   "训练泄漏"的误判；进度打印（每 10 批）与池诊断对齐后才看清两段斜率。
5. **升级框架时先跑 T=1 冒烟**：0.18→0.21 的 drop 语义变化在 T=1 时完全无感
   （激活量小），T=4 才引爆；显存压力与"激活数 × 停留时长"成正比，任何惰性化
   （lazy drop / 延迟释放）改动都要用最大 T 验证。

## 12. 附：关键文件清单

```
patches/burn-fusion/            # 本地补丁：ContinueDrop→drain（核心修复）+ drop 计数器
patches/cubecl-runtime/         # 诊断补丁：池直方图 / pinned_ids / alloc_sizes 反查
patches/../Cargo.toml           # [patch.crates-io] 指向上述两个目录（勿删）
rust-sdt/src/train.rs           # 逐批 sync、eval 内层后端、to_wgpu_weights、mem-probe、进度打印
rust-sdt/src/main.rs            # MemProbe 子命令
rust-sdt/cubecl.toml            # memory.logger 已置 disabled（诊断时临时改 full）
rust-sdt/probe_vram.ps1         # VRAM 轮询探针
rust-sdt/artifacts/train_burn.csv      # 修复后训练结果（可复现）
rust-sdt/artifacts/train_report.txt    # train-compare 判定报告
```
