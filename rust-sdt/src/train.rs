//! Burn 训练循环：小规模 CIFAR-10 训练，验证架构迁移误差（Task 5）。
//!
//! 实现要点：
//! - 参数化：`TrainSdt`/`TrainConv` 用 `#[derive(Module)]` 把权重包成 `Param`，
//!   前向复用 `crate::model::forward_full`（只读 Tensor 的泛型实现），
//!   每步通过 `to_weights()` 把 Param 值转为只读权重视图。
//! - 优化器：`burn::optim::SgdConfig`（momentum=0.9、dampening=0、无 weight decay），
//!   语义与 PyTorch `torch.optim.SGD(lr, momentum=0.9)` 一致。
//! - 静态校准：NPZ 融合权重基于初始 BN running stats（mu=0, var=1），与真实激活分布
//!   失配会导致 LIF 全网零脉冲死锁；用 512 张训练图对每个「输出喂给 LIF 的层」做
//!   逐通道 (mu, sigma) 统计并做等效替换（W' <- W'/sigma, B' <- (B'-mu)/sigma），
//!   与 PyTorch 侧 scripts/train_pytorch_reference.py 的校准严格对齐。
//! - 损失：`burn::nn::loss::CrossEntropyLoss`（logits + Int 标签）。
//! - 评估：测试集分批推理，logits `.inner()` 转无梯度张量后 argmax 统计 top-1。
//! - 输出：每 epoch 打印 train_loss / val_top1，并追加写 artifacts/train_burn.csv。

use burn::module::Param;
use burn::optim::{GradientsParams, SgdConfig};
use burn::prelude::*;

use crate::config::SdtConfig;
use crate::loader::{Cifar10Npz, Split};
use crate::model::{forward_full, BackendAdapter, Dev, SdtWeights};

/// 训练用 autodiff 后端：Wgpu 的自动微分包装（burn 0.18 需开启 autodiff feature）
pub type AutodiffBackend = burn::backend::Autodiff<BackendAdapter>;

/// 逐批 sync 开关（环境变量 `SDT_BATCH_SYNC`，默认开启；置 0 可复现 burn 0.21 的
/// 首个 step 显存爆炸，用于 A/B 对照）。
///
/// 原理：burn-fusion 0.21 MultiStream 中普通张量 drop（ContinueDrop）只把 Drop op
/// 入队而不触发 drain，Drop op 真正执行（`ExecutionMode::Sync` 全量执行 +
/// `drain_queue` 对 ReadWrite 节点 `handles.free()`）后缓冲才归还 cubecl 内存池。
/// T=4 一步前向+反向产生数千个中间张量，若不逐批排空，死张量缓冲在整个 step 内
/// 持续累积（实测首个 step +16GB，24GB RTX 4090 触顶 OOM）；逐批 sync 强制排空，
/// 让下一步复用池中缓冲而非新开大页。
fn batch_sync_enabled() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| std::env::var("SDT_BATCH_SYNC").map(|v| v != "0").unwrap_or(true))
}

/// 逐批 memory_cleanup 开关（环境变量 `SDT_BATCH_CLEANUP`，默认关闭）。
///
/// sync 后已死缓冲归还池内空闲分片，可被后续分配复用；但 cubecl 0.10 SubSlices
/// 池的空闲分页只有显式 cleanup 才释放回 GPU。若逐批 sync 后显存仍缓慢爬升
/// （碎片化导致复用失败），置 1 可每批释放全空分页（开销略增）。
fn batch_cleanup_enabled() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| std::env::var("SDT_BATCH_CLEANUP").map(|v| v == "1").unwrap_or(false))
}
/// autodiff 后端对应的设备类型（burn 0.21 中 Autodiff<B>::Device == B::Device，与 wgpu 设备同型）
pub type AutodiffDevice = <AutodiffBackend as burn::tensor::backend::BackendTypes>::Device;

/// 训练用单个卷积层参数（可训练）：[out, in, kH, kW] 与 [out]
#[derive(Module, Debug)]
pub struct TrainConv<B: Backend> {
    /// 卷积核权重 [out, in, kH, kW]
    pub w: Param<Tensor<B, 4>>,
    /// 卷积偏置 [out]
    pub b: Param<Tensor<B, 1>>,
}

/// 训练用单个 block 的参数（可训练）
#[derive(Module, Debug)]
pub struct TrainBlock<B: Backend> {
    /// SSA 的 q 投影
    pub q: TrainConv<B>,
    /// SSA 的 k 投影
    pub k: TrainConv<B>,
    /// SSA 的 v 投影
    pub v: TrainConv<B>,
    /// talking_heads 权重（PyTorch 前向未使用；Tensor 实现常量 Module，不参与训练）
    pub th_w: Tensor<B, 2>,
    /// SSA 输出投影
    pub proj: TrainConv<B>,
    /// MLP 第一层
    pub fc1: TrainConv<B>,
    /// MLP 第二层
    pub fc2: TrainConv<B>,
}

/// 训练用完整模型参数（BN 已融合，与 SdtWeights 键名一一对应）
#[derive(Module, Debug)]
pub struct TrainSdt<B: Backend> {
    /// SPS 4 级卷积
    pub pe_proj: [TrainConv<B>; 4],
    /// SPS rpe 投影
    pub pe_rpe: TrainConv<B>,
    /// Transformer blocks
    pub blk: Vec<TrainBlock<B>>,
    /// 分类头权重 [10, 256]
    pub head_w: Param<Tensor<B, 2>>,
    /// 分类头偏置 [10]
    pub head_b: Param<Tensor<B, 1>>,
}

/// 把单个可训练卷积层参数转为只读权重视图
fn conv_to_weights<B: Backend>(c: &TrainConv<B>) -> crate::model::ConvLayer<B> {
    crate::model::ConvLayer {
        w: c.w.val(),
        b: c.b.val(),
    }
}

impl<B: Backend> TrainSdt<B> {
    /// 转为模型前向所需的只读权重集合（Param 值的克隆）。
    /// `forward_full` 只读 Tensor，因此每步前向前调用本方法即可复用统一前向。
    pub fn to_weights(&self) -> SdtWeights<B> {
        SdtWeights {
            pe_proj: [
                conv_to_weights(&self.pe_proj[0]),
                conv_to_weights(&self.pe_proj[1]),
                conv_to_weights(&self.pe_proj[2]),
                conv_to_weights(&self.pe_proj[3]),
            ],
            pe_rpe: conv_to_weights(&self.pe_rpe),
            blk: self
                .blk
                .iter()
                .enumerate()
                .map(|(j, b)| crate::model::BlockWeights {
                    q: conv_to_weights(&b.q),
                    k: conv_to_weights(&b.k),
                    v: conv_to_weights(&b.v),
                    // talking_heads 前向不使用，克隆占位以保持结构一致
                    th_w: b.th_w.clone(),
                    proj: conv_to_weights(&b.proj),
                    fc1: conv_to_weights(&b.fc1),
                    fc2: conv_to_weights(&b.fc2),
                    // block 序号取 Vec 下标（taps/save_init_npz 键名依赖）
                    layer: j,
                })
                .collect(),
            head_w: self.head_w.val(),
            head_b: self.head_b.val(),
        }
    }

    /// 从只读权重集合构造可训练模型（Param 包裹并开启梯度）。
    /// 权重张量已在指定设备上，`Param::from_tensor` 会自动标记 require_grad。
    pub fn from_weights(w: &SdtWeights<B>) -> Self {
        let conv = |c: &crate::model::ConvLayer<B>| TrainConv::<B> {
            w: Param::from_tensor(c.w.clone()),
            b: Param::from_tensor(c.b.clone()),
        };
        TrainSdt::<B> {
            pe_proj: [
                conv(&w.pe_proj[0]),
                conv(&w.pe_proj[1]),
                conv(&w.pe_proj[2]),
                conv(&w.pe_proj[3]),
            ],
            pe_rpe: conv(&w.pe_rpe),
            blk: w
                .blk
                .iter()
                .map(|b| TrainBlock::<B> {
                    q: conv(&b.q),
                    k: conv(&b.k),
                    v: conv(&b.v),
                    // talking_heads 不参与训练，直接持有张量
                    th_w: b.th_w.clone(),
                    proj: conv(&b.proj),
                    fc1: conv(&b.fc1),
                    fc2: conv(&b.fc2),
                })
                .collect(),
            head_w: Param::from_tensor(w.head_w.clone()),
            head_b: Param::from_tensor(w.head_b.clone()),
        }
    }
}

/// 训练超参数（CLI 传入）
pub struct TrainArgs {
    /// 训练配置名称（当前仅支持 cifar10_s，即 SdtConfig::default()）
    pub preset: String,
    /// 训练轮数
    pub epochs: u32,
    /// 数据目录（优先在其中查找 cifar10_data.npz，否则回退 artifacts/）
    pub data_dir: String,
    /// 随机种子（控制初始化与数据混洗）
    pub seed: u64,
    /// 初始化权重 NPZ 路径（None 表示按分布随机初始化）
    pub weights: Option<String>,
    /// 批大小
    pub batch_size: usize,
    /// 学习率
    pub lr: f64,
    /// 时间步数 T（静态帧重复次数）
    pub time_steps: usize,
    /// 是否跳过静态校准（false = 默认校准，与 PyTorch 基准对齐）
    pub no_calibrate: bool,
}

/// 按截断正态分布（std=0.02，与 PyTorch trunc_normal 一致）初始化 [out, in, kH, kW] 卷积权重
fn rand_conv_w4<R: rand::Rng>(rng: &mut R, shape: [usize; 4], device: &Dev) -> Tensor<BackendAdapter, 4> {
    let normal = rand_distr::Normal::new(0.0, 0.02).expect("正态分布参数无效");
    let data: Vec<f32> = (0..shape[0] * shape[1] * shape[2] * shape[3])
        .map(|_| rng.sample(normal) as f32)
        .collect();
    Tensor::<BackendAdapter, 1>::from_floats(data.as_slice(), device)
        .reshape(shape)
        // 截断到 [-2*std, 2*std]（PyTorch trunc_normal 的默认截断）
        .clamp(4.0f32.mul_add(-0.02, 0.0), 0.02f32.mul_add(4.0, 0.0))
}

/// 按截断正态分布初始化 [out, in] 线性权重
fn rand_lin_w2<R: rand::Rng>(rng: &mut R, shape: [usize; 2], device: &Dev) -> Tensor<BackendAdapter, 2> {
    let normal = rand_distr::Normal::new(0.0, 0.02).expect("正态分布参数无效");
    let data: Vec<f32> = (0..shape[0] * shape[1])
        .map(|_| rng.sample(normal) as f32)
        .collect();
    Tensor::<BackendAdapter, 1>::from_floats(data.as_slice(), device)
        .reshape(shape)
        .clamp(4.0f32.mul_add(-0.02, 0.0), 0.02f32.mul_add(4.0, 0.0))
}

/// 按形状生成全零张量（bias 初始化为 0）
fn zeros<const D2: usize>(shape: [usize; D2], device: &Dev) -> Tensor<BackendAdapter, D2> {
    Tensor::zeros(shape, device)
}

/// 按同分布随机初始化构建权重集合（PyTorch 侧 trunc_normal(std=0.02) 的 Rust 近似）。
/// 各层权重形状严格对齐 load_weights 从 NPZ 读到的形状：
/// - SPS: proj0/1 = 3x3（in_ch -> ch），proj2 = 3x3（ch -> 2*ch），proj3 = 3x3（2ch -> 4ch），
///   rpe = 3x3（4ch -> 4ch）；q/k/v/proj = 1x1（4ch -> 4ch）；
///   fc1 = 1x1（4ch -> 4*mlp_ratio*ch），fc2 = 1x1（4*mlp_ratio*ch -> 4ch）
/// - head: Linear(4ch, num_classes)
///
/// 借用冲突说明：随机层构造函数以 `&mut StdRng` 传参（普通函数而非闭包），
/// 顺序消费同一个 rng，避免 E0499（闭包捕获 &mut rng 后又可变借用）与
/// E0596（FnMut 闭包需要 mut 绑定）的借用冲突。
/// 3x3 卷积层（截断正态权重 + 零偏置）
fn rand_conv_layer(rng: &mut rand::rngs::StdRng, in_ch: usize, out_ch: usize, k: usize, device: &Dev) -> crate::model::ConvLayer<BackendAdapter> {
    crate::model::ConvLayer {
        w: rand_conv_w4(rng, [out_ch, in_ch, k, k], device),
        b: zeros([out_ch], device),
    }
}

/// 1x1 卷积层（截断正态权重 + 零偏置）
fn rand_conv1x1_layer(rng: &mut rand::rngs::StdRng, in_ch: usize, out_ch: usize, device: &Dev) -> crate::model::ConvLayer<BackendAdapter> {
    rand_conv_layer(rng, in_ch, out_ch, 1, device)
}

fn init_weights(cfg: &SdtConfig, device: &Dev, seed: u64) -> SdtWeights<BackendAdapter> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let ch = cfg.embed_dims / 4; // SPS 每级的通道增长基数（256 -> 64）
    let mlp_hidden = ch * 4 * cfg.mlp_ratio; // 顶层 dim * mlp_ratio = 1024

    let mut blk = Vec::with_capacity(cfg.depths);
    for j in 0..cfg.depths {
        blk.push(crate::model::BlockWeights {
            q: rand_conv1x1_layer(&mut rng, 4 * ch, 4 * ch, device),
            k: rand_conv1x1_layer(&mut rng, 4 * ch, 4 * ch, device),
            v: rand_conv1x1_layer(&mut rng, 4 * ch, 4 * ch, device),
            // talking_heads 前向不使用，占位为零
            th_w: zeros([4 * ch, 4 * ch], device),
            proj: rand_conv1x1_layer(&mut rng, 4 * ch, 4 * ch, device),
            fc1: rand_conv1x1_layer(&mut rng, 4 * ch, mlp_hidden, device),
            fc2: rand_conv1x1_layer(&mut rng, mlp_hidden, 4 * ch, device),
            // block 序号（用于 taps/save_init_npz 键名）
            layer: j,
        });
    }

    SdtWeights {
        pe_proj: [
            rand_conv_layer(&mut rng, cfg.in_channels, ch, 3, device),
            rand_conv_layer(&mut rng, ch, 2 * ch, 3, device),
            rand_conv_layer(&mut rng, 2 * ch, 4 * ch, 3, device),
            rand_conv_layer(&mut rng, 4 * ch, 4 * ch, 3, device),
        ],
        pe_rpe: rand_conv_layer(&mut rng, 4 * ch, 4 * ch, 3, device),
        blk,
        head_w: rand_lin_w2(&mut rng, [cfg.num_classes, 4 * ch], device),
        head_b: zeros([cfg.num_classes], device),
    }
}

/// 校准层选择：枚举前向路径上所有「输出直接喂给 LIF 的层」+ head（与 PyTorch layer_list 一致）
#[derive(Clone, Copy, PartialEq)]
enum LayerSel {
    /// SPS 第 i 级 conv
    PeProj(usize),
    /// SPS rpe conv
    PeRpe,
    /// block j 的第 which 个 conv：0=q 1=k 2=v 3=proj 4=fc1 5=fc2
    Blk { j: usize, which: usize },
    /// 分类头（Linear）
    Head,
}

/// block 层的 taps 键名（与 forward_taps / PyTorch 导出键一致）
fn blk_key(j: usize, which: usize) -> &'static str {
    match (j, which) {
        (0, 0) => "blk0_q",
        (0, 1) => "blk0_k",
        (0, 2) => "blk0_v",
        (0, 3) => "blk0_proj",
        (0, 4) => "blk0_fc1",
        (0, 5) => "blk0_fc2",
        (1, 0) => "blk1_q",
        (1, 1) => "blk1_k",
        (1, 2) => "blk1_v",
        (1, 3) => "blk1_proj",
        (1, 4) => "blk1_fc1",
        (1, 5) => "blk1_fc2",
        _ => panic!("暂不支持的 block 数量（depths > 2）"),
    }
}

/// 返回校准层清单（顺序与 PyTorch calibrate_fused_weights 的 layer_list 严格一致）
fn calib_layer_list(depths: usize) -> Vec<(LayerSel, &'static str)> {
    let mut list = vec![
        (LayerSel::PeProj(0), "pe_proj0"),
        (LayerSel::PeProj(1), "pe_proj1"),
        (LayerSel::PeProj(2), "pe_proj2"),
        (LayerSel::PeProj(3), "pe_proj3"),
        (LayerSel::PeRpe, "pe_rpe"),
    ];
    for j in 0..depths {
        for which in 0..6 {
            list.push((LayerSel::Blk { j, which }, blk_key(j, which)));
        }
    }
    list.push((LayerSel::Head, "head"));
    list
}

/// 按层选择返回可变卷积层引用（head 返回 None，由调用方单独处理）
fn pick_conv<'a>(
    w: &'a mut SdtWeights<BackendAdapter>,
    sel: LayerSel,
) -> Option<&'a mut crate::model::ConvLayer<BackendAdapter>> {
    match sel {
        LayerSel::PeProj(i) => Some(&mut w.pe_proj[i]),
        LayerSel::PeRpe => Some(&mut w.pe_rpe),
        LayerSel::Blk { j, which } => {
            let b = &mut w.blk[j];
            match which {
                0 => Some(&mut b.q),
                1 => Some(&mut b.k),
                2 => Some(&mut b.v),
                3 => Some(&mut b.proj),
                4 => Some(&mut b.fc1),
                _ => Some(&mut b.fc2),
            }
        }
        LayerSel::Head => None,
    }
}

/// 对「输出直接喂给 LIF 的层」（键名与 forward_taps 一致）统计其输出的逐通道 (mu, sigma)。
///
/// 统计方式与 PyTorch calibrate_fused_weights 的 tap 钩子一致：把该层输出按通道外维度
/// 展平成 [N, C] 后求总体方差（unbiased=False），sigma = sqrt(var + 1e-6)。
/// 每层单独重跑全部 16 批前向，以复现 PyTorch 的「逐层级联」语义：
/// 上游层统计完成后立即替换权重（其 LIF 被激活），下游层统计基于已激活的激活值。
fn layer_output_stats(
    w: &SdtWeights<BackendAdapter>,
    data: &Cifar10Npz,
    indices: &[usize],
    calib_bs: usize,
    t: usize,
    device: &Dev,
    cfg: &SdtConfig,
    key: &str,
) -> (Tensor<BackendAdapter, 1>, Tensor<BackendAdapter, 1>) {
    // 跨批累计「逐通道 mu / sigma」的平均（各批样本数相同）
    let mut sum: Option<Tensor<BackendAdapter, 1>> = None;
    let mut sumsq: Option<Tensor<BackendAdapter, 1>> = None;

    for chunk in indices.chunks(calib_bs) {
        let (images, _) = data.get_batch(Split::Train, chunk, t, device);
        let taps = crate::model::forward_taps(images, w, cfg);
        let ten = taps
            .map
            .get(key)
            .unwrap_or_else(|| panic!("taps 缺少校准层 {}", key));
        // 统一 5 维布局 [T, B, C, ...]：通道维是 dim2，其余合并进样本维
        let dims = ten.dims();
        let c = dims[2];
        let rest: usize = dims[3..].iter().product::<usize>().max(1);
        // [T, B, C, rest] -> [T*B, C, rest] -> [T*B, rest, C] -> [T*B*rest, C]
        // 注意：必须交换 dim1/dim2（把通道移到最后一维）后再 reshape，
        // 这样 reshape 后每列才对应一个通道（旧实现交换 dim0/dim2 后 reshape
        // 会打乱通道归属，算出的是「全体 std」而非「逐通道 std」）。
        let flat = ten
            .clone()
            .reshape([dims[0] * dims[1], c, rest]);
        let flat = flat
            .swap_dims(1, 2)
            .reshape([rest * dims[0] * dims[1], c]);
        // ---- 两遍方差算法（数值稳定）----
        // 与 PyTorch out.var(unbiased=False) 一致：先算 mu，再算 E[(x-mu)^2]。
        // 旧单遍算法（sq/n - mu^2）在均值主导方差时（mu >> sigma）会发生
        // f32 灾难性抵消（conv 输出 mu~0.1、sigma~0.001，sigma 被高估 2 倍以上）。
        let n = (rest * dims[0] * dims[1]) as f32;
        // mu：sum_dim 保持维数（[1, C]），squeeze 回 [C]
        let mu = flat.clone().sum_dim(0).squeeze::<1>().div_scalar(n); // [C]
        // centered：flat - mu（广播 [N,C] - [1,C]）
        let centered = flat - mu.clone().unsqueeze::<2>();
        // var = E[(x-mu)^2)；sigma = sqrt(var + 1e-6)
        let sq = centered
            .clone()
            .mul(centered)
            .sum_dim(0)
            .squeeze::<1>()
            .div_scalar(n); // [C]
        let sigma_c = (sq + 1e-6f32).sqrt();
        // sum_dim 保持维数（[1, C]），squeeze 回 [C]
        sum = Some(match sum {
            Some(e) => e + mu.clone(),
            None => mu.clone(),
        });
        sumsq = Some(match sumsq {
            Some(e) => e + sigma_c.clone(),
            None => sigma_c.clone(),
        });

        // 逐批 sync+cleanup（顺序约束与 train_epoch 相同）：校准是纯前向循环，
        // 每批前向激活（T=4,B=32 下 ~50 个 8MB 张量）若不逐批归还，2720 次前向
        // 的累积在 24GB 卡上会挤占后续训练的显存空间（实测校准后残留 4.7~6.8GB）。
        <crate::model::BackendAdapter as burn::tensor::backend::Backend>::sync(device)
            .expect("校准逐批同步失败");
        <crate::model::BackendAdapter as burn::tensor::backend::Backend>::memory_cleanup(device);
    }

    // 跨批平均（各批样本数相同，直接平均）
    let n_batches = (indices.len().div_ceil(calib_bs)) as f32;
    let mu = sum.expect("校准统计缺失").div_scalar(n_batches);
    let sigma = sumsq.expect("校准统计缺失").div_scalar(n_batches);
    (mu, sigma)
}

/// 静态校准：对每个「输出直接喂给 LIF 的层」测量真实数据下输出逐通道 (mu, sigma)，
/// 然后做等效替换 W' <- W'/sigma、B' <- (B'-mu)/sigma（等价于用真实统计量重新融合 BN）。
///
/// 层清单与统计方式严格对齐 PyTorch scripts/train_pytorch_reference.py 的
/// `calibrate_fused_weights`：pe_proj0..3、pe_rpe、每 block 的 q/k/v/proj/fc1/fc2，
/// 最后是 head（Linear，统计在其 [T,B,C] 输出上展开为 [T*B, C] 进行）。
///
/// 关键语义（与 PyTorch 严格一致）：轮内「逐层」处理——每统计完一层立即替换其权重，
/// 下一层的统计基于「上游已替换、LIF 已激活」的前向。若一轮内一次前向全统计（旧实现），
/// 上游 LIF 死锁期间下游 conv 输出退化为常数偏置（var≈0），替换时权重会爆炸发散。
pub fn static_calibrate(
    weights: &SdtWeights<BackendAdapter>,
    data: &Cifar10Npz,
    device: &Dev,
    cfg: &SdtConfig,
) -> SdtWeights<BackendAdapter> {
    // 校准超参数：与 PyTorch 侧一致（16 批 x 32 = 512 张训练图，最多 10 轮，容差 1e-3）
    let num_batches = 16usize;
    let calib_bs = 32usize;
    let max_rounds = 10usize;
    let tol = 1e-3f64;
    let t = cfg.time_steps;

    // 构造校准输入（训练集前 num_batches*calib_bs 张，固定顺序无随机性）
    let n_calib = (num_batches * calib_bs).min(data.n_train);
    let indices: Vec<usize> = (0..n_calib).collect();

    // 逐层级联校准清单（顺序 = 前向顺序 = PyTorch layer_list）
    let layer_list = calib_layer_list(cfg.depths);

    // 当前工作副本（每轮就地替换）
    let mut w = weights.clone();

    for round in 1..=max_rounds {
        // 本轮报告：(键名, 该层「通道平均」sigma)——口径与 PyTorch float(sigma.mean()) 一致
        let mut report: Vec<(String, f64)> = Vec::new();

        for (sel, key) in &layer_list {
            // ---- 统计该层输出的逐通道 (mu, sigma)（基于当前权重的前向）----
            let (mu, sigma) = layer_output_stats(
                &w, data, &indices, calib_bs, t, device, cfg, key,
            );
            let sig_v: Vec<f32> = sigma
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .expect("读取 sigma 失败");
            let sig_mean = sig_v.iter().sum::<f32>() / sig_v.len() as f32;
            // ---- 立即替换该层权重：W <- W/sigma, B <- (B-mu)/sigma ----
            match sel {
                LayerSel::Head => {
                    // Linear 等效替换：W <- W / sigma[:, None]（按输出通道缩放）
                    let sig2 = sigma.clone().reshape([sig_v.len(), 1]);
                    w.head_w = w.head_w.clone().div(sig2);
                    w.head_b = (w.head_b.clone() - mu).div(sigma);
                }
                _ => {
                    // conv 等效替换：w <- w / sigma[:, None, None, None]，b <- (b - mu) / sigma
                    let layer = pick_conv(&mut w, *sel).expect("校准层选择无效");
                    let sig4 = sigma.clone().reshape([sig_v.len(), 1, 1, 1]);
                    layer.w = layer.w.clone().div(sig4);
                    layer.b = (layer.b.clone() - mu).div(sigma);
                }
            }
            report.push((key.to_string(), sig_mean as f64));
        }

        // 逐层明细打印（排查振荡层：哪层 sigma 远离 1）
        let detail = report
            .iter()
            .map(|(k, m)| format!("{}={:.3}", k, m))
            .collect::<Vec<_>>()
            .join(" ");
        println!("[calibrate] 逐层 sigma: {}", detail);

        // 收敛判断：各层「平均 sigma」与 1 的最大偏差 < tol（与 PyTorch 一致）
        let err = report
            .iter()
            .map(|(_, m)| (m - 1.0).abs())
            .fold(0f64, f64::max);
        println!(
            "[calibrate] 第 {} 轮: max|sigma-1|={:.4}（层平均 sigma 范围 {:.3}~{:.3}）",
            round, err,
            report.iter().map(|(_, m)| *m).fold(f64::INFINITY, f64::min),
            report
                .iter()
                .map(|(_, m)| *m)
                .fold(f64::NEG_INFINITY, f64::max)
        );
        if err < tol {
            println!("[calibrate] 已收敛（{} 轮）", round);
            break;
        }
    }

    w
}

/// 单个 epoch 的训练：返回（按样本数加权的平均损失，步数）
///
/// `model` 以 `&mut` 传入：burn 0.18 的优化器 `step` 消费旧模块并返回更新后的新模块，
/// 每步通过 `*model = optim.step(...)` 就地替换。
/// 优化器参数用泛型 `O: Optimizer<...>`：SgdConfig::init 返回的
/// `simple::adaptor::OptimizerAdaptor` 未被公开 re-export（E0603），
/// 但 `burn::optim::Optimizer` trait 是公开的，按 trait 泛型传入即可解耦具体类型。
fn train_epoch<O>(
    model: &mut TrainSdt<AutodiffBackend>,
    optim: &mut O,
    data: &Cifar10Npz,
    order: &[usize],
    batch_size: usize,
    t: usize,
    lr: f64,
    device: &AutodiffDevice,
    cfg: &SdtConfig,
) -> (f64, usize)
where
    O: burn::optim::Optimizer<
        TrainSdt<AutodiffBackend>,
        AutodiffBackend,
    >,
{
    // 交叉熵（均值损失，与 PyTorch CrossEntropyLoss 默认一致）
    let loss_fn = burn::nn::loss::CrossEntropyLoss::new(None, device);
    let mut loss_sum = 0f64;
    let mut n_samples = 0usize;
    let epoch_start = std::time::Instant::now();

    for chunk in order.chunks(batch_size) {
        // 组装批次：图像 [T,B,3,32,32]，标签 [B]（Int）
        let (images, targets) = data.get_batch(Split::Train, chunk, t, device);
        let bs = chunk.len();

        // 前向（复用统一前向；to_weights 产出 autodiff 后端的只读权重视图）
        let weights = model.to_weights();
        let logits = forward_full(images, &weights, cfg, None);

        // 交叉熵损失（[B] -> 取均值得到标量）
        let loss = loss_fn.forward(logits, targets);
        // 标量读取：into_scalar() 是 Tensor 公开 API（primitive 为私有字段，不可直接访问）
        let loss_scalar = loss.clone().into_scalar() as f64;

        // backward + SGD 更新（Tensor::backward 为公开 API；
        // burn 0.18 优化器消费旧模型返回新模型）
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, model);
        *model = optim.step(lr, model.clone(), grads);

        // 逐批强制 fusion drain（burn 0.21 显存修复，详见 batch_sync_enabled 注释）：
        // 每步 optim.step 后排空队列，让本批全部已死中间张量的 Drop op 真正执行、
        // GPU 缓冲归还内存池，下一步复用而非新开大页。顺序约束与 epoch 末相同：
        // 只 sync 不 cleanup（cleanup 仅在显式启用 SDT_BATCH_CLEANUP=1 时执行）。
        if batch_sync_enabled() {
            <AutodiffBackend as burn::tensor::backend::Backend>::sync(&AutodiffDevice::default())
                .expect("GPU 逐批同步失败");
        }
        if batch_cleanup_enabled() {
            <AutodiffBackend as burn::tensor::backend::Backend>::memory_cleanup(
                &AutodiffDevice::default(),
            );
        }

        // 累计加权损失
        loss_sum += loss_scalar * bs as f64;
        n_samples += bs;

        // 进度打印（每 10 批；便于对照 cubecl 池诊断时间线定位泄漏窗口）
        let batch_idx = n_samples / batch_size;
        if batch_idx % 10 == 0 {
            println!(
                "[train-progress] batch {batch_idx} loss={loss_scalar:.4} t={:.1}s",
                epoch_start.elapsed().as_secs_f32()
            );
        }
    }

    // 每个 epoch 结束后显式同步 GPU 并触发内存池清理（顺序不可颠倒）：
    // 1) sync：阻塞等待 fusion 队列中全部 GPU 命令完成并清空队列；
    // 2) memory_cleanup：stream 空闲后释放 cubecl 内存池（SubSlices 策略）中
    //    已无引用的中间缓冲区分片。
    // 若跳过 sync 直接 cleanup，会与 fusion 后台执行线程并发访问句柄容器，
    // 触发 "Should have handle for tensor" 竞态 panic（实测复现）。
    // 若只 sync 不 cleanup，空闲分片持续累积，第二轮显存占用与耗时暴涨
    // （观测：epoch1=75s、epoch2=676s）。
    <AutodiffBackend as burn::tensor::backend::Backend>::sync(&AutodiffDevice::default())
        .expect("GPU 同步失败");
    <AutodiffBackend as burn::tensor::backend::Backend>::memory_cleanup(&AutodiffDevice::default());

    let steps = order.len().div_ceil(batch_size);
    (loss_sum / n_samples.max(1) as f64, steps)
}

/// 测试集评估：返回 top-1 准确率（百分数）
///
/// 推理在**内层 wgpu 后端**（无 autodiff 图）上运行：autodiff 前向注册的图节点
/// 在无 backward 消费时不释放，会把每批全部激活（~50 个 8MB 张量）钉在显存里
/// （实测 62 批 eval 累积 ~26GB → OOM）。转回内层后端后中间张量正常 drop 复用。
fn eval_top1(
    model: &TrainSdt<AutodiffBackend>,
    data: &Cifar10Npz,
    batch_size: usize,
    t: usize,
    device: &AutodiffDevice,
    cfg: &SdtConfig,
) -> f64 {
    // SDT_EVAL_BACKEND=inner（默认，无图不泄漏）| autodiff（旧路径，对照实验用）
    let eval_backend = std::env::var("SDT_EVAL_BACKEND").unwrap_or_else(|_| "inner".to_string());
    let weights_wgpu = to_wgpu_weights(&model.to_weights());
    let n = data.n_test;
    let order: Vec<usize> = (0..n).collect();
    let mut correct = 0usize;
    let mut eval_batch = 0usize;
    let eval_start = std::time::Instant::now();

    for chunk in order.chunks(batch_size) {
        let (logits, targets) = if eval_backend == "autodiff" {
            let (images, targets) = data.get_batch(Split::Test, chunk, t, device);
            let weights = model.to_weights();
            let logits = forward_full(images, &weights, cfg, None);
            let logits = logits.detach().inner();
            let targets = targets.inner();
            (logits, targets)
        } else {
            let (images, targets) = data.get_batch(Split::Test, chunk, t, wgpu_dev());
            let logits = forward_full(images, &weights_wgpu, cfg, None);
            (logits, targets)
        };
        let pred = logits.argmax(1);
        let pred_v: Vec<i64> = pred
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("读取预测结果失败");
        let tgt_v: Vec<i64> = targets
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("读取标签失败");
        for (p, y) in pred_v.iter().zip(tgt_v.iter()) {
            if p == y {
                correct += 1;
            }
        }

        // 逐批 sync+cleanup（顺序约束与 train_epoch 相同）：及时归还全空页
        <AutodiffBackend as burn::tensor::backend::Backend>::sync(device)
            .expect("eval 逐批同步失败");
        <AutodiffBackend as burn::tensor::backend::Backend>::memory_cleanup(device);

        eval_batch += 1;
        if eval_batch % 10 == 0 {
            println!(
                "[eval-progress] batch {eval_batch} correct={correct} t={:.1}s",
                eval_start.elapsed().as_secs_f32()
            );
        }
    }

    // 评估结束后同样先 sync 再清理内存池（与 train_epoch 相同的顺序约束）
    <AutodiffBackend as burn::tensor::backend::Backend>::sync(device)
        .expect("GPU 同步失败");
    <AutodiffBackend as burn::tensor::backend::Backend>::memory_cleanup(device);

    correct as f64 / n as f64 * 100.0
}

/// 旧签名兼容入口（默认不校准？否——保持默认校准语义，见 run_train_with_args）
// main.rs 现直接调用 run_train_with_args，本入口保留供旧调用方兼容
#[allow(dead_code)]
pub fn run_train(preset: &str, epochs: u32, data_dir: &str, seed: u64) {
    // 默认超参数（保持旧签名兼容；CLI 扩展参数走 run_train_with_args）
    run_train_with_args(TrainArgs {
        preset: preset.to_string(),
        epochs,
        data_dir: data_dir.to_string(),
        seed,
        weights: None,
        batch_size: 32,
        lr: 0.01,
        time_steps: 4,
        no_calibrate: false,
    });
}

/// 带完整参数的训练入口（main.rs 的 train 子命令调用）
pub fn run_train_with_args(args: TrainArgs) {
    // preset 校验：当前仅支持默认配置
    if args.preset != "cifar10_s" {
        eprintln!("错误：暂不支持 preset = {}（当前仅支持 cifar10_s）", args.preset);
        std::process::exit(1);
    }
    let cfg = SdtConfig::default();

    // 数据路径：优先 data_dir 下的 cifar10_data.npz，否则回退 artifacts/
    let mut data_path = format!("{}/cifar10_data.npz", args.data_dir.trim_end_matches(['/', '\\']));
    if !std::path::Path::new(&data_path).exists() {
        data_path = "artifacts/cifar10_data.npz".to_string();
    }
    println!("== Burn CIFAR-10 小规模训练 ==");
    println!(
        "数据: {}（train={} test={}）batch={} lr={} T={} calibrate={}",
        data_path, "8000", "2000", args.batch_size, args.lr, args.time_steps, !args.no_calibrate
    );
    let data = Cifar10Npz::load(&data_path);

    // 设备与计时
    let device: AutodiffDevice = Default::default();
    let start = std::time::Instant::now();

    // 权重初始化：加载 NPZ（BN 融合形式）或按同分布随机初始化
    // 权重在 wgpu 后端构造；训练前统一 detach 并搬到 autodiff 后端（设备同型，搬运无拷贝开销）
    let mut init_weights_final: SdtWeights<BackendAdapter> = match &args.weights {
        Some(w) if !w.is_empty() => {
            println!("加载初始化权重: {}", w);
            let npz = crate::tensor_io::read_npz(w);
            // load_weights 需要 cfg 以确定 block 数量等结构信息
            crate::model::load_weights(&npz, &cfg, wgpu_dev())
        }
        _ => {
            println!("按 trunc_normal(std=0.02) 随机初始化（seed={}）", args.seed);
            init_weights(&cfg, wgpu_dev(), args.seed)
        }
    };

    // 静态校准（默认开启；--no-calibrate 可关）：与 PyTorch 基准对齐的关键步骤
    if !args.no_calibrate {
        let t0 = std::time::Instant::now();
        init_weights_final = static_calibrate(&init_weights_final, &data, wgpu_dev(), &cfg);
        println!(
            "[calibrate] 静态校准完成（耗时 {:.1}s）",
            t0.elapsed().as_secs_f32()
        );
        // 把校准后的初始权重按 NPZ 键名导出（便于复现与排查）
        save_init_npz(&init_weights_final, "artifacts/train_burn_init.npz");
    }

    // 校准/加载结束后，训练开始前同步 GPU 并回收内存池（顺序不可颠倒，与 epoch 结束相同）：
    // 校准阶段约 2720 次前向（17 层 x 10 轮 x 16 批）之间只在每层统计末尾读一次数据，
    // 大量中间张量的 drop 操作被惰性积压；cubecl 内存池（SubSlices 策略）的
    // dealloc_period 为 None，空闲分页从不自动回收。若不清理，首个训练 step 的
    // 前向+反向图将无显存可分（实测 24GB RTX 4090 上触发上万次 wgpu Out of Memory，
    // 并连锁导致 "Should have handle for tensor" panic）。
    // 1) sync：阻塞等待 GPU 队列排空，使积压的 drop 全部执行、句柄释放回内存池；
    // 2) memory_cleanup：回收池中已无引用的空闲分页，消除碎片化占用。
    <AutodiffBackend as burn::tensor::backend::Backend>::sync(&AutodiffDevice::default())
        .expect("GPU 同步失败");
    <AutodiffBackend as burn::tensor::backend::Backend>::memory_cleanup(&AutodiffDevice::default());

    // 构造可训练模型（Param 包裹，自动 require_grad）
    // from_weights 需要 autodiff 后端权重：把 wgpu 权重 detach 后包装为 autodiff 后端张量
    // （burn 0.18 中 Autodiff<B> 的张量由 TensorPrimitive::Float 包裹，用 Tensor::from_inner 构造）
    let init_weights_ad: SdtWeights<AutodiffBackend> = to_autodiff_weights(&init_weights_final);
    let mut model = TrainSdt::<AutodiffBackend>::from_weights(&init_weights_ad);
    println!("可训练参数量: {}", model.num_params());

    // SGD 优化器：momentum=0.9、dampening=0（与 PyTorch torch.optim.SGD(lr, momentum=0.9) 一致）
    // SgdConfig::init 在 AutodiffBackend（Autodiff<Wgpu>）上实例化，返回 simple::adaptor 包装
    // with_momentum 的参数是 Option<MomentumConfig>（None 表示关闭动量）
    let mut optim = SgdConfig::new()
        .with_momentum(Some(burn::optim::momentum::MomentumConfig {
            momentum: 0.9,
            dampening: 0.0,
            nesterov: false,
        }))
        .init::<AutodiffBackend, TrainSdt<AutodiffBackend>>();

    // 每 epoch 的数据顺序（种子随 epoch 变化，保证可复现且每轮不同）
    let t = args.time_steps;
    let mut csv_rows: Vec<String> = vec!["epoch,train_loss,val_top1".to_string()];

    for epoch in 0..args.epochs {
        let ep_start = std::time::Instant::now();
        let order = crate::loader::shuffled_indices(data.n_train, args.seed + epoch as u64);
        let (train_loss, _steps) = train_epoch(
            &mut model,
            &mut optim,
            &data,
            &order,
            args.batch_size,
            t,
            args.lr,
            &device,
            &cfg,
        );
        let val_top1 = eval_top1(&model, &data, args.batch_size, t, &device, &cfg);
        println!(
            "epoch={}/{}, train_loss={:.6}, val_top1={:.2}%（本轮耗时 {:.1}s）",
            epoch + 1,
            args.epochs,
            train_loss,
            val_top1,
            ep_start.elapsed().as_secs_f32()
        );
        csv_rows.push(format!("{},{:.6},{:.2}", epoch + 1, train_loss, val_top1));
    }

    // 写 CSV（覆盖式）：artifacts/train_burn.csv
    let csv_path = "artifacts/train_burn.csv";
    if let Some(dir) = std::path::Path::new(csv_path).parent() {
        std::fs::create_dir_all(dir).expect("创建 artifacts 目录失败");
    }
    std::fs::write(csv_path, csv_rows.join("\n") + "\n").expect("写入 train_burn.csv 失败");
    println!(
        "训练指标已写入: {}（总耗时 {:.1}s）",
        csv_path,
        start.elapsed().as_secs_f32()
    );
}

/// 把 autodiff 后端的权重集合转回内层 wgpu 后端（`Tensor::inner`，`to_autodiff_weights` 的镜像）。
///
/// eval/推理应在无 autodiff 图的内层后端上运行：autodiff 前向会为全部中间张量
/// 注册图节点，而无 backward 消费时这些节点（及其 GPU 缓冲句柄）不会被释放
/// ——实测 62 批 eval 累积 ~26GB 激活直至 OOM。转回内层后端后，中间张量
/// 走普通 drop 路径（复用正常）。
fn to_wgpu_weights(w: &SdtWeights<AutodiffBackend>) -> SdtWeights<BackendAdapter> {
    fn conv_wgpu(c: &crate::model::ConvLayer<AutodiffBackend>) -> crate::model::ConvLayer<BackendAdapter> {
        crate::model::ConvLayer {
            w: c.w.clone().inner(),
            b: c.b.clone().inner(),
        }
    }
    SdtWeights {
        pe_proj: [
            conv_wgpu(&w.pe_proj[0]),
            conv_wgpu(&w.pe_proj[1]),
            conv_wgpu(&w.pe_proj[2]),
            conv_wgpu(&w.pe_proj[3]),
        ],
        pe_rpe: conv_wgpu(&w.pe_rpe),
        blk: w
            .blk
            .iter()
            .map(|b| crate::model::BlockWeights {
                q: conv_wgpu(&b.q),
                k: conv_wgpu(&b.k),
                v: conv_wgpu(&b.v),
                th_w: b.th_w.clone().inner(),
                proj: conv_wgpu(&b.proj),
                fc1: conv_wgpu(&b.fc1),
                fc2: conv_wgpu(&b.fc2),
                layer: b.layer,
            })
            .collect(),
        head_w: w.head_w.clone().inner(),
        head_b: w.head_b.clone().inner(),
    }
}

/// 把 wgpu 后端的权重集合转为 autodiff 后端（Autodiff<Wgpu>）。
///
/// burn 0.18 中 `Autodiff<B>` 的张量可用 `Tensor::from_inner`（autodiff 扩展的公开方法）
/// 从 inner 后端（wgpu）张量构造；设备同型（Autodiff<B>::Device == B::Device），
/// 张量数据不发生拷贝，仅包装为 autodiff 叶节点。
fn to_autodiff_weights(w: &SdtWeights<BackendAdapter>) -> SdtWeights<AutodiffBackend> {
    // 单个卷积层：逐张量转换
    fn conv_ad(c: &crate::model::ConvLayer<BackendAdapter>) -> crate::model::ConvLayer<AutodiffBackend> {
        crate::model::ConvLayer {
            w: burn::tensor::Tensor::from_inner(c.w.clone()),
            b: burn::tensor::Tensor::from_inner(c.b.clone()),
        }
    }
    SdtWeights {
        pe_proj: [
            conv_ad(&w.pe_proj[0]),
            conv_ad(&w.pe_proj[1]),
            conv_ad(&w.pe_proj[2]),
            conv_ad(&w.pe_proj[3]),
        ],
        pe_rpe: conv_ad(&w.pe_rpe),
        blk: w
            .blk
            .iter()
            .map(|b| crate::model::BlockWeights {
                q: conv_ad(&b.q),
                k: conv_ad(&b.k),
                v: conv_ad(&b.v),
                // talking_heads 前向不使用，同样转换以保持结构一致
                th_w: burn::tensor::Tensor::from_inner(b.th_w.clone()),
                proj: conv_ad(&b.proj),
                fc1: conv_ad(&b.fc1),
                fc2: conv_ad(&b.fc2),
                layer: b.layer,
            })
            .collect(),
        head_w: burn::tensor::Tensor::from_inner(w.head_w.clone()),
        head_b: burn::tensor::Tensor::from_inner(w.head_b.clone()),
    }
}

/// 把校准后的初始权重按 NPZ 键名导出（键名与 sdt_reference.npz 的权重键一致）
fn save_init_npz(w: &SdtWeights<BackendAdapter>, path: &str) {
    use crate::tensor_io::NpyArray;
    let mut arrays: Vec<NpyArray> = Vec::new();
    for (i, layer) in w.pe_proj.iter().enumerate() {
        arrays.push(NpyArray::from_tensor4(&layer.w, &format!("pe_proj{}_w", i)));
        arrays.push(NpyArray::from_tensor1(&layer.b, &format!("pe_proj{}_b", i)));
    }
    arrays.push(NpyArray::from_tensor4(&w.pe_rpe.w, "pe_rpe_w"));
    arrays.push(NpyArray::from_tensor1(&w.pe_rpe.b, "pe_rpe_b"));
    for b in &w.blk {
        let j = b.layer;
        arrays.push(NpyArray::from_tensor4(&b.q.w, &format!("blk{}_q_w", j)));
        arrays.push(NpyArray::from_tensor1(&b.q.b, &format!("blk{}_q_b", j)));
        arrays.push(NpyArray::from_tensor4(&b.k.w, &format!("blk{}_k_w", j)));
        arrays.push(NpyArray::from_tensor1(&b.k.b, &format!("blk{}_k_b", j)));
        arrays.push(NpyArray::from_tensor4(&b.v.w, &format!("blk{}_v_w", j)));
        arrays.push(NpyArray::from_tensor1(&b.v.b, &format!("blk{}_v_b", j)));
        // talking_heads 权重（PyTorch 导出的 NPZ 含 blk{j}_th_w；校准不改动它，原样写出）
        arrays.push(NpyArray::from_tensor2(&b.th_w, &format!("blk{}_th_w", j)));
        arrays.push(NpyArray::from_tensor4(&b.proj.w, &format!("blk{}_proj_w", j)));
        arrays.push(NpyArray::from_tensor1(&b.proj.b, &format!("blk{}_proj_b", j)));
        arrays.push(NpyArray::from_tensor4(&b.fc1.w, &format!("blk{}_fc1_w", j)));
        arrays.push(NpyArray::from_tensor1(&b.fc1.b, &format!("blk{}_fc1_b", j)));
        arrays.push(NpyArray::from_tensor4(&b.fc2.w, &format!("blk{}_fc2_w", j)));
        arrays.push(NpyArray::from_tensor1(&b.fc2.b, &format!("blk{}_fc2_b", j)));
    }
    arrays.push(NpyArray::from_tensor2(&w.head_w, "head_w"));
    arrays.push(NpyArray::from_tensor1(&w.head_b, "head_b"));
    crate::tensor_io::write_npz(path, &arrays);
    println!("校准后初始权重已写入: {}", path);
}

/// 获取 wgpu 设备引用（张量构造使用 wgpu 设备，训练在 Autodiff<Wgpu> 上进行，两者同构）
fn wgpu_dev() -> &'static Dev {
    // 设备为轻量结构，用 OnceLock 造 'static 引用（每次调用返回同一默认设备）
    static ONCE: std::sync::OnceLock<Dev> = std::sync::OnceLock::new();
    ONCE.get_or_init(Default::default)
}

/// 解析训练指标 CSV（header + 若干行 "epoch,train_loss,val_top1"），返回 (epoch, loss, top1) 列表
fn parse_train_csv(path: &str) -> Vec<(u32, f64, f64)> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("读取 {} 失败: {}", path, e));
    let mut rows = Vec::new();
    for line in text.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert!(parts.len() >= 3, "{} 行格式错误: {}", path, line);
        rows.push((
            parts[0].trim().parse().expect("epoch 解析失败"),
            parts[1].trim().parse().expect("train_loss 解析失败"),
            parts[2].trim().parse().expect("val_top1 解析失败"),
        ));
    }
    assert!(!rows.is_empty(), "{} 没有数据行", path);
    rows
}

/// 训练结果对照入口（main.rs 的 train-compare 子命令调用）
///
/// 读取 Burn 与 PyTorch 两份训练 CSV，各取最后一行（最终 epoch），判定口径（2026-09-08 定）：
/// - 最终 val top-1 **相对偏差** ≤ top1_tol%（分母 = pytorch 值；长跑下 top1 量级
///   从 10%→65% 变化，相对差跨阶段可比）
/// - 最终 train loss **绝对差** ≤ loss_tol（深度过拟合区 loss 分母小，相对差噪声
///   被口径放大——实测 100 epoch 绝对差 0.031 / 相对差 19.3%，故 loss 用绝对差）
/// 两条件同时满足判 PASS，报告写入 artifacts/train_report.txt 并同步输出到 stdout。
pub fn run_train_compare(burn_csv: &str, pytorch_csv: &str, top1_tol: f64, loss_tol: f64) {
    println!("== Burn vs PyTorch 训练结果对照 ==");
    let burn = parse_train_csv(burn_csv);
    let pt = parse_train_csv(pytorch_csv);

    // 各取最后一个 epoch（两份 CSV 的 epoch 数应一致，否则取各自最后并警告）
    let (be, bl, bt) = *burn.last().expect("burn csv 为空");
    let (pe, pl, pt1) = *pt.last().expect("pytorch csv 为空");
    if burn.len() != pt.len() {
        println!(
            "警告：两份 CSV 的 epoch 数不同（burn={} pytorch={}），取各自最后一行比较",
            burn.len(),
            pt.len()
        );
    }

    // 判定：top1 相对偏差（%）、loss 绝对差（分母=pytorch 值防除零）
    let top1_diff = (bt - pt1).abs();
    let loss_diff = (bl - pl).abs();
    let top1_rel = top1_diff / pt1.abs().max(1e-12) * 100.0;
    let top1_ok = top1_rel <= top1_tol;
    let loss_ok = loss_diff <= loss_tol;
    let all_ok = top1_ok && loss_ok;

    // 报告内容
    let mut lines: Vec<String> = Vec::new();
    lines.push("== 训练结果对照报告 ==".to_string());
    lines.push(format!("burn_csv: {}", burn_csv));
    lines.push(format!("pytorch_csv: {}", pytorch_csv));
    lines.push(format!(
        "判定口径：val_top1 相对偏差 ≤{top1_tol}%（分母=pytorch）；train_loss 绝对差 ≤{loss_tol}"
    ));
    lines.push(String::new());
    lines.push("-- 逐 epoch 指标 --".to_string());
    lines.push("epoch | burn: train_loss / val_top1% | pytorch: train_loss / val_top1%".to_string());
    let n = burn.len().max(pt.len());
    for i in 0..n {
        let b = burn.get(i).copied();
        let p = pt.get(i).copied();
        lines.push(format!(
            "{} | {} | {}",
            i + 1,
            b.map_or("-".into(), |(_, l, t)| format!("{:.6} / {:.2}", l, t)),
            p.map_or("-".into(), |(_, l, t)| format!("{:.6} / {:.2}", l, t)),
        ));
    }
    lines.push(String::new());
    lines.push(format!(
        "-- 最终判定（burn epoch={} vs pytorch epoch={}） --",
        be, pe
    ));
    lines.push(format!(
        "val_top1: burn={:.2}% pytorch={:.2}% 绝对差={:.2}pp 相对差={:.2}% 允许=≤{:.2}% 判定={}",
        bt, pt1, top1_diff, top1_rel, top1_tol,
        if top1_ok { "PASS" } else { "FAIL" }
    ));
    lines.push(format!(
        "train_loss: burn={:.6} pytorch={:.6} 绝对差={:.6} 允许=≤{:.2} 判定={}",
        bl, pl, loss_diff, loss_tol,
        if loss_ok { "PASS" } else { "FAIL" }
    ));
    lines.push(format!(
        "最终判定: {}",
        if all_ok { "PASS" } else { "FAIL" }
    ));

    // 终端输出 + 写文件
    for l in &lines {
        println!("{}", l);
    }
    let report_path = "artifacts/train_report.txt";
    if let Some(dir) = std::path::Path::new(report_path).parent() {
        std::fs::create_dir_all(dir).expect("创建 artifacts 目录失败");
    }
    std::fs::write(report_path, lines.join("\n") + "\n").expect("写入 train_report.txt 失败");
    println!("报告已写入: {}", report_path);

    // 判定失败时以非零码退出，便于脚本化检查
    if !all_ok {
        std::process::exit(1);
    }
}

/// 显存最小复现（`cargo run -- mem-probe`）：
/// - iters>0：简单 conv 循环（栈层基线）
/// - stage：用真实模型走真实训练内环（dummy 数据），逐步加入部件做二分：
///   1=仅前向+标量读；2=+backward；3=+SGD step（完整内环）
/// 配合 BURN_SDT_POOL_DIAG=1 判断泄漏在栈层还是模型/优化器层
pub fn run_mem_probe(iters: u32) {
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;

    let stage: u32 = std::env::var("SDT_PROBE_STAGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if stage == 0 {
        // —— 栈层基线：纯 conv fwd+bwd（已验证复用正常）——
        let dev = AutodiffDevice::default();
        println!("== mem-probe: {iters} iters of conv[128,256,8,8] fwd+bwd ==");
        let w: Tensor<AutodiffBackend, 4> = Tensor::zeros([256, 256, 1, 1], &dev).require_grad();
        let x0: Tensor<AutodiffBackend, 4> = Tensor::zeros([128, 256, 8, 8], &dev);
        for i in 0..iters {
            let x = x0.clone();
            let y = burn::tensor::module::conv2d(
                x,
                w.clone(),
                None,
                burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            );
            let yr = burn::tensor::activation::relu(y.clone());
            let z = y.clone() + yr;
            let loss = z.clone().mul_scalar(2.0).sum();
            let _ = loss.clone().into_scalar();
            let grads = loss.backward();
            drop(grads);
            if i % 50 == 0 {
                <AutodiffBackend as Backend>::sync(&dev).ok();
                println!("[probe] iter {i}");
            }
        }
        <AutodiffBackend as Backend>::sync(&dev).ok();
        println!("== mem-probe done ==");
        return;
    }

    // —— 真实模型内环二分：dummy 数据 + 真实 forward_full + CE + （可选）SGD ——
    let cfg = SdtConfig::default();
    let device: AutodiffDevice = Default::default();
    let cfg_wgpu = cfg.clone();
    println!("== mem-probe: real-model stage={stage}, {iters} iters ==");

    // 随机初始化权重（结构与 NPZ 一致；无需校准）；NPZ 模式在下方覆盖
    let cfg_wgpu = cfg.clone();
    println!("== mem-probe: real-model stage={stage}, {iters} iters ==");

    let mut optim = SgdConfig::new()
        .with_momentum(Some(burn::optim::momentum::MomentumConfig {
            momentum: 0.9,
            dampening: 0.0,
            nesterov: false,
        }))
        .init::<AutodiffBackend, TrainSdt<AutodiffBackend>>();

    let loss_fn = burn::nn::loss::CrossEntropyLoss::new(None, &device);
    let t = 4usize;
    let b = 32usize;

    // SDT_PROBE_REALDATA=1：用真实 NPZ 数据 + get_batch 组批（write 路径），
    // 否则用 zeros（kernel fill 路径）——二分创建路径的泄漏
    // SDT_PROBE_NPZWEIGHTS=1：完全复刻 train 命令的权重路径（read_npz + load_weights）
    // stage=4：直接调用 train_epoch 本体（256 批 dummy order），彻底消除实现差异
    let real_data = std::env::var("SDT_PROBE_REALDATA").map(|v| v == "1").unwrap_or(false);
    let data = if real_data {
        Some(crate::loader::Cifar10Npz::load("artifacts/cifar10_data.npz"))
    } else {
        None
    };

    let npz_weights = std::env::var("SDT_PROBE_NPZWEIGHTS").map(|v| v == "1").unwrap_or(false);
    let weights_ad: SdtWeights<AutodiffBackend> = if npz_weights {
        let npz = crate::tensor_io::read_npz("artifacts/sdt_reference.npz");
        let w_wgpu = crate::model::load_weights(&npz, &cfg, wgpu_dev());
        let ad = to_autodiff_weights(&w_wgpu);
        // SDT_PROBE_KEEPWGPU=1：模拟 run_train_with_args 中 init_weights_final 全程存活
        if std::env::var("SDT_PROBE_KEEPWGPU").map(|v| v == "1").unwrap_or(false) {
            std::mem::forget(w_wgpu);
            println!("[probe] wgpu weights kept alive (forget)");
        }
        ad
    } else {
        to_autodiff_weights(&init_weights(&cfg_wgpu, wgpu_dev(), 42))
    };
    let mut model = TrainSdt::<AutodiffBackend>::from_weights(&weights_ad);

    if stage == 4 {
        // 直接调用真实 train_epoch：模型 + SGD + 真实数据 + 逐批 sync，全走生产代码
        let data = data.expect("stage4 需要 SDT_PROBE_REALDATA=1");
        // 复刻 run_train_with_args 的训练前 sync+cleanup（泄漏触发嫌疑 #1）
        if std::env::var("SDT_PROBE_PRECLEAN").map(|v| v == "1").unwrap_or(false) {
            <AutodiffBackend as Backend>::sync(&device).expect("probe sync 失败");
            <AutodiffBackend as Backend>::memory_cleanup(&device);
            println!("[probe] pre-train sync+cleanup done");
        }
        let mut optim = SgdConfig::new()
            .with_momentum(Some(burn::optim::momentum::MomentumConfig {
                momentum: 0.9,
                dampening: 0.0,
                nesterov: false,
            }))
            .init::<AutodiffBackend, TrainSdt<AutodiffBackend>>();
        let order: Vec<usize> = (0..iters as usize * b).collect();
        let (loss, steps) = train_epoch(
            &mut model, &mut optim, &data, &order, b, t, 0.01, &device, &cfg,
        );
        println!("== stage4 train_epoch done: loss={loss:.4} steps={steps} ==");
        return;
    }

    if stage == 5 {
        // 进程内直接调用 run_train_with_args：若泄漏复现，则差异在它的 setup/epoch 结构里
        println!("== stage5: calling run_train_with_args in-process ==");
        run_train_with_args(crate::train::TrainArgs {
            preset: "cifar10_s".to_string(),
            epochs: 1,
            data_dir: "data/cifar10".to_string(),
            seed: 42,
            weights: Some("artifacts/sdt_reference.npz".to_string()),
            batch_size: b,
            lr: 0.01,
            time_steps: t,
            no_calibrate: true,
        });
        println!("== stage5 done ==");
        return;
    }

    for i in 0..iters {
        // dummy 批次：图像 [T,B,3,32,32]、标签 [B]
        let (images, targets) = if let Some(d) = &data {
            let lo = (i as usize * b) % (d.n_train - b);
            let idx: Vec<usize> = (lo..lo + b).collect();
            d.get_batch(crate::loader::Split::Train, &idx, t, &device)
        } else {
            let images = Tensor::<AutodiffBackend, 5>::zeros([t as usize, b, 3, 32, 32], &device)
                .require_grad();
            let targets = Tensor::<AutodiffBackend, 1, burn::tensor::Int>::zeros([b], &device);
            (images, targets)
        };

        let weights = model.to_weights();
        let logits = forward_full(images, &weights, &cfg, None);

        let loss = loss_fn.forward(logits, targets);
        let _ = loss.clone().into_scalar();

        if stage >= 2 {
            let grads = loss.backward();
            if stage >= 3 {
                let gp = GradientsParams::from_grads(grads, &model);
                model = burn::optim::Optimizer::step(&mut optim, 0.01, model.clone(), gp);
            } else {
                drop(grads);
            }
        }

        // 与真实 train_epoch 一致：逐批 sync（SDT_PROBE_SYNC=1 时；默认每 20 迭代）
        let sync_every = std::env::var("SDT_PROBE_SYNC")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(20);
        if i % sync_every == 0 {
            <AutodiffBackend as Backend>::sync(&device).ok();
            println!("[probe] iter {i} (stage {stage})");
        }
    }
    <AutodiffBackend as Backend>::sync(&device).ok();
    println!("== mem-probe done (stage {stage}) ==");
}
