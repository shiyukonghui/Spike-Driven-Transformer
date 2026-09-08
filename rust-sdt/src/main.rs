//! Spike-Driven Transformer 的 Rust + Burn 实现入口。
//!
//! 子命令：
//! - `export`（Python 侧脚本，非本程序）：导出 PyTorch 权重与中间张量到 NPZ
//! - `forward-check`：加载权重，运行 Burn 前向推理，与 PyTorch 输出进行误差对照
//! - `train`：使用 Burn 完成小规模 CIFAR-10 训练，验证架构迁移误差在允许范围内

// 深度类型递归限制：burn Module derive 会为嵌套模型（TrainSdt 含数组/Vec 嵌套 Param）
// 生成深层 trait 求值链，默认 128 会触发 E0275 溢出（如 num_params 的 Send/Sync 证明链）
#![recursion_limit = "512"]

mod check;
mod config;
mod es_train;
mod loader;
mod model;
mod ops;
mod train;
mod tensor_io;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rust-sdt", about = "Spike-Driven Transformer 的 Rust + Burn 实现")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 前向误差对照：Burn 与 PyTorch 输出的相对误差统计
    ForwardCheck {
        /// PyTorch 导出的 NPZ 文件（含权重与中间张量）
        #[arg(long, default_value = "artifacts/sdt_reference.npz")]
        npz: String,
        /// 随机种子
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// 训练结果对照：读取 Burn 与 PyTorch 两份训练 CSV，判定迁移误差是否在允许范围
    TrainCompare {
        /// Burn 侧训练指标 CSV
        #[arg(long, default_value = "artifacts/train_burn.csv")]
        burn_csv: String,
        /// PyTorch 基准训练指标 CSV
        #[arg(long, default_value = "artifacts/train_pytorch.csv")]
        pytorch_csv: String,
        /// 最终 val top-1 允许相对偏差（%，分母=pytorch 值）
        #[arg(long, default_value_t = 5.0)]
        top1_tol: f64,
        /// 最终 train loss 允许绝对差
        #[arg(long, default_value_t = 0.1)]
        loss_tol: f64,
    },
    /// 小规模训练对照：Burn 训练数个 epoch，与 PyTorch 同步对照
    Train {
        /// 训练配置名称（对应 config.rs 中的预设）
        #[arg(long, default_value = "cifar10_s")]
        preset: String,
        /// 训练轮数
        #[arg(long, default_value_t = 2)]
        epochs: u32,
        /// 数据目录（CIFAR-10 npz 解包目录）
        #[arg(long, default_value = "data/cifar10")]
        data_dir: String,
        /// 随机种子
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// 初始化权重 NPZ 路径（默认 artifacts/sdt_reference.npz，与 PyTorch 基准对齐）
        #[arg(long, default_value = "artifacts/sdt_reference.npz")]
        weights: String,
        /// 批大小
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        /// 学习率
        #[arg(long, default_value_t = 0.01)]
        lr: f64,
        /// 时间步数 T（静态帧重复次数）
        #[arg(long, default_value_t = 4)]
        time_steps: usize,
        /// 跳过静态校准（默认校准，与 PyTorch 基准对齐）
        #[arg(long, default_value_t = false)]
        no_calibrate: bool,
    },
    /// 显存最小复现：反复对 [128,256,8,8]（8MB）张量做 conv+backward+drop，
    /// 配合 BURN_SDT_POOL_DIAG=1 判断泄漏在栈层还是模型层
    MemProbe {
        /// 迭代次数
        #[arg(long, default_value_t = 300)]
        iters: u32,
    },
    /// 演化策略（EggRoll-ES，无反向传播）训练：LoRA 低秩扰动 + loglik 奖励 +
    /// z-score + AdamW，全程内层 wgpu 后端（详见 es_train.rs 模块注释）
    TrainEs {
        /// 训练轮数（1 epoch = 全部训练图过一遍）
        #[arg(long, default_value_t = 100)]
        epochs: u32,
        /// 数据目录（CIFAR-10 npz 解包目录）
        #[arg(long, default_value = "data/cifar10")]
        data_dir: String,
        /// 随机种子
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// 初始化权重 NPZ 路径（默认 artifacts/sdt_reference.npz，与 SGD 基线同源）
        #[arg(long, default_value = "artifacts/sdt_reference.npz")]
        weights: String,
        /// 时间步数 T（静态帧重复次数）
        #[arg(long, default_value_t = 4)]
        time_steps: usize,
        /// 跳过静态校准
        #[arg(long, default_value_t = false)]
        no_calibrate: bool,
        /// 每代候选数（偶数，候选成对反对称；factored 模式下 = 噪声方向数 = 图数）
        #[arg(long, default_value_t = 8000)]
        pop: usize,
        /// 每候选评估的图像块大小（cache 模式：块内共享扰动；factored 模式固定 1）
        #[arg(long, default_value_t = 1)]
        es_batch: usize,
        /// 实现模式：factored（默认，因式分解噪声前向，SPS 冻结）| cache（ΔW 物化缓存）
        #[arg(long, default_value = "factored")]
        mode: String,
        /// factored 模式每 chunk 的候选（图）数
        #[arg(long, default_value_t = 128)]
        chunk: usize,
        /// 扰动幅度 σ（相对值：每槽实际幅度 = σ × 该层权重 std）
        #[arg(long, default_value_t = 0.5)]
        sigma: f32,
        /// LoRA 秩 r
        #[arg(long, default_value_t = 32)]
        rank: usize,
        /// AdamW 学习率
        #[arg(long, default_value_t = 0.01)]
        lr: f64,
        /// 每 N 个 epoch 评估一次验证集
        #[arg(long, default_value_t = 1)]
        validate_every: u32,
        /// CSV 输出路径
        #[arg(long, default_value = "artifacts/train_burn_es.csv")]
        csv_out: String,
        /// TSES 温度松弛：ES fitness 前向用 σ(β(h−thr)) 松弛 LIF（ES_MANIFOLD_NOTE 定理 3）
        #[arg(long, default_value_t = false)]
        relax: bool,
        /// 松弛温度 β（= STE α；4.0 与 spikingjelly 代理一致）
        #[arg(long, default_value_t = 4.0)]
        beta: f32,
        /// β 退火周期：每 N 个 epoch β×2（上限 16）；0 = 固定 β
        #[arg(long, default_value_t = 0)]
        beta_anneal_every: u32,
        /// S6 松弛作用域：all（全部位点，v4 行为）| ssa（仅 SSA 位点 xs/q/k/v/kv，MLP/head 硬 LIF）
        #[arg(long, default_value = "all")]
        relax_scope: String,
        /// 可变学习率：线性 warmup 的 epoch 数（0 = 恒定 lr）
        #[arg(long, default_value_t = 0)]
        lr_warmup: u32,
        /// 可变学习率：余弦退火终值比例（lr_min = lr × 此值）
        #[arg(long, default_value_t = 0.0)]
        lr_min_frac: f64,
        /// S4（opt-in）：β≥16 时训练 fitness 切硬 LIF（默认关：全程松弛）
        #[arg(long, default_value_t = false)]
        hard_at_16: bool,
        /// QAM 调制（ES_MANIFOLD §12）："off" | "learnable"（m=(1+a)·cos(φ)）
        #[arg(long, default_value = "off")]
        qam: String,
        /// QAM 位点集："head"（feat，位点深度律最高信号）| "ssa" | "all"
        #[arg(long, default_value = "head")]
        qam_sites: String,
        /// ± 对共享同一样本（协议修正：对偶差分消去数据难度项）
        #[arg(long, default_value_t = false)]
        pair_shared: bool,
        /// 块结构消融：conv（默认）| none（恒等直连）| probe（纯线性探针）
        #[arg(long, default_value = "conv")]
        blocks: String,
        /// 冻结卷积权重槽（σ=0，仅 conv 模式有效）
        #[arg(long, default_value_t = false)]
        freeze_lora: bool,
        /// 逐候选零噪声锚点（控制变量 X1）：raw′=raw−b₀(x) 再 z-score
        #[arg(long, default_value_t = false)]
        baseline_zero: bool,
        /// 逐槽步长自适应（X3，NES/CSA 式漂移幅值 EMA）
        #[arg(long, default_value_t = false)]
        sigma_adapt: bool,
        /// σ 自适应周期（epoch）
        #[arg(long, default_value_t = 10)]
        sigma_adapt_every: u32,
        /// 温缩 z-score（X2-lite）：候选 winsorize 到上一 epoch 分布的 ±3σ
        #[arg(long, default_value_t = false)]
        robust_z: bool,
        /// hard 0/1 适应度（原库默认）：argmax==标签 替代 loglik
        #[arg(long, default_value_t = false)]
        fitness_hard: bool,
    },
    /// 混合估计器（ES_MANIFOLD §13）：SGD（W/b 精确梯度）+ ES（v_th 8 标量 + QAM 低维槽）
    TrainMixed {
        /// 训练轮数
        #[arg(long, default_value_t = 100)]
        epochs: u32,
        /// 数据目录（含 cifar10_data.npz）
        #[arg(long, default_value = "data")]
        data_dir: String,
        #[arg(long, default_value_t = 79)]
        seed: u64,
        /// 初始化权重 NPZ（校准后 init）
        #[arg(long)]
        weights: Option<String>,
        #[arg(long, default_value_t = 4)]
        time_steps: usize,
        #[arg(long, default_value_t = false)]
        no_calibrate: bool,
        /// SGD 批大小
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        /// SGD 学习率（cosine 到 0.1×）
        #[arg(long, default_value_t = 0.01)]
        lr_sgd: f64,
        #[arg(long, default_value_t = 0.1)]
        lr_sgd_min_frac: f64,
        /// SGD 动量
        #[arg(long, default_value_t = 0.9)]
        sgd_momentum: f64,
        /// ES 候选数（= 训练集）
        #[arg(long, default_value_t = 8000)]
        es_pop: usize,
        #[arg(long, default_value_t = 128)]
        es_chunk: usize,
        /// ES 扰动基准尺度
        #[arg(long, default_value_t = 0.5)]
        sigma_es: f32,
        /// ES 更新学习率
        #[arg(long, default_value_t = 0.02)]
        lr_es: f32,
        #[arg(long, default_value_t = 4.0)]
        beta: f32,
        /// β 退火周期（×2，上限 16）；0 = 固定
        #[arg(long, default_value_t = 25)]
        beta_anneal_every: u32,
        /// QAM："off" | "learnable"
        #[arg(long, default_value = "learnable")]
        qam: String,
        /// QAM 位点："head" | "ssa" | "all"
        #[arg(long, default_value = "ssa")]
        qam_sites: String,
        #[arg(long, default_value_t = 5)]
        validate_every: u32,
        #[arg(long, default_value = "artifacts/train_mixed.csv")]
        csv_out: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::ForwardCheck { npz, seed } => {
            crate::check::run_forward_check(&npz, seed);
        }
        Command::TrainCompare {
            burn_csv,
            pytorch_csv,
            top1_tol,
            loss_tol,
        } => {
            // 训练结果对照：两份 CSV 各取最后一行比较，写入 train_report.txt
            crate::train::run_train_compare(&burn_csv, &pytorch_csv, top1_tol, loss_tol);
        }
        Command::Train {
            preset,
            epochs,
            data_dir,
            seed,
            weights,
            batch_size,
            lr,
            time_steps,
            no_calibrate,
        } => {
            // 完整 CLI 参数透传到训练入口（含 --weights/--batch-size/--lr/--time-steps/--no-calibrate）
            crate::train::run_train_with_args(crate::train::TrainArgs {
                preset,
                epochs,
                data_dir,
                seed,
                weights: Some(weights),
                batch_size,
                lr,
                time_steps,
                no_calibrate,
            });
        }
        Command::MemProbe { iters } => {
            crate::train::run_mem_probe(iters);
        }
        Command::TrainEs {
            epochs,
            data_dir,
            seed,
            weights,
            time_steps,
            no_calibrate,
            pop,
            es_batch,
            sigma,
            rank,
            lr,
            validate_every,
            csv_out,
            mode,
            chunk,
            relax,
            beta,
            beta_anneal_every,
            relax_scope,
            lr_warmup,
            lr_min_frac,
            hard_at_16,
            qam,
            qam_sites,
            pair_shared,
            blocks,
            freeze_lora,
            baseline_zero,
            sigma_adapt,
            sigma_adapt_every,
            robust_z,
            fitness_hard,
        } => {
            crate::es_train::run_train_es(crate::es_train::EsArgs {
                epochs,
                data_dir,
                seed,
                weights: Some(weights),
                time_steps,
                no_calibrate,
                pop,
                es_batch,
                sigma,
                rank,
                lr,
                validate_every,
                csv_out,
                mode,
                chunk,
                relax,
                beta,
                beta_anneal_every,
                relax_scope,
                lr_warmup,
                lr_min_frac,
                hard_at_16,
                qam,
                qam_sites,
                pair_shared,
                blocks,
                freeze_lora,
                baseline_zero,
                sigma_adapt,
                sigma_adapt_every,
                robust_z,
                fitness_hard,
            });
        }
        Command::TrainMixed {
            epochs,
            data_dir,
            seed,
            weights,
            time_steps,
            no_calibrate,
            batch_size,
            lr_sgd,
            lr_sgd_min_frac,
            sgd_momentum,
            es_pop,
            es_chunk,
            sigma_es,
            lr_es,
            beta,
            beta_anneal_every,
            qam,
            qam_sites,
            validate_every,
            csv_out,
        } => {
            crate::es_train::run_train_mixed(crate::es_train::MixedArgs {
                epochs,
                data_dir,
                seed,
                weights,
                time_steps,
                no_calibrate,
                batch_size,
                lr_sgd,
                lr_sgd_min_frac,
                sgd_momentum,
                es_pop,
                es_chunk,
                sigma_es,
                lr_es,
                beta,
                beta_anneal_every,
                qam,
                qam_sites,
                validate_every,
                csv_out,
            });
        }
    }
}
