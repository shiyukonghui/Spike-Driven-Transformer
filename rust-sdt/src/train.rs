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
/// autodiff 后端对应的设备类型（burn 0.18 中 Autodiff<B>::Device == B::Device，与 wgpu 设备同型）
pub type AutodiffDevice = <AutodiffBackend as Backend>::Device;

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
        let mu = flat.clone().sum_dim(0).squeeze::<1>(0).div_scalar(n); // [C]
        // centered：flat - mu（广播 [N,C] - [1,C]）
        let centered = flat - mu.clone().unsqueeze::<2>();
        // var = E[(x-mu)^2)；sigma = sqrt(var + 1e-6)
        let sq = centered
            .clone()
            .mul(centered)
            .sum_dim(0)
            .squeeze::<1>(0)
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

    for chunk in order.chunks(batch_size) {
        // 组装批次：图像 [T,B,3,32,32]，标签 [B]（Int）
        let (images, targets) = data.get_batch(Split::Train, chunk, t, device);
        let bs = chunk.len();

        // 前向（复用统一前向；to_weights 产出 autodiff 后端的只读权重视图）
        let weights = model.to_weights();
        let logits = forward_full(images, &weights, cfg);

        // 交叉熵损失（[B] -> 取均值得到标量）
        let loss = loss_fn.forward(logits, targets);
        // 标量读取：into_scalar() 是 Tensor 公开 API（primitive 为私有字段，不可直接访问）
        let loss_scalar = loss.clone().into_scalar() as f64;

        // backward + SGD 更新（Tensor::backward 为公开 API；
        // burn 0.18 优化器消费旧模型返回新模型）
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, model);
        *model = optim.step(lr, model.clone(), grads);

        // 累计加权损失
        loss_sum += loss_scalar * bs as f64;
        n_samples += bs;
    }
    let steps = order.len().div_ceil(batch_size);
    (loss_sum / n_samples.max(1) as f64, steps)
}

/// 测试集评估：返回 top-1 准确率（百分数）
fn eval_top1(
    model: &TrainSdt<AutodiffBackend>,
    data: &Cifar10Npz,
    batch_size: usize,
    t: usize,
    device: &AutodiffDevice,
    cfg: &SdtConfig,
) -> f64 {
    let n = data.n_test;
    let order: Vec<usize> = (0..n).collect();
    let mut correct = 0usize;

    for chunk in order.chunks(batch_size) {
        let (images, targets) = data.get_batch(Split::Test, chunk, t, device);
        let weights = model.to_weights();
        let logits = forward_full(images, &weights, cfg);
        // 转无梯度张量后 argmax（等价 no_grad 推理；detach 为 Tensor 公开 API）
        let logits = logits.detach();
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
    }
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
