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
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::ForwardCheck { npz, seed } => {
            crate::check::run_forward_check(&npz, seed);
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
    }
}
