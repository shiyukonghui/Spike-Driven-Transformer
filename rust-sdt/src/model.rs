//! Burn 版 SpikeDrivenTransformer 前向实现（BN 已融合，推理语义；前向泛型化支持训练后端）。
//!
//! 模块结构（与 PyTorch 侧 model/spikeformer.py + module/ 一一对应）：
//! - MS_SPS：4 级 Conv3x3+LIF（"0011"：前两级不池化、后两级 maxpool3x3/2）+ rpe 残差
//! - MS_Block_Conv × depths：MS_SSA_Conv（线性脉冲注意力）+ MS_MLP_Conv
//! - head：flatten(3).mean(3) -> LIF -> Linear
//!
//! 张量布局约定：内部统一使用 [T, B, C, H, W]（时间维在前，与 PyTorch 实现一致），
//! 卷积在 [T*B, C, H, W] 上执行。
//!
//! 泛型说明：训练（Task 5）要求在 autodiff 后端上反传梯度，因此全部前向函数
//! 与权重结构体均泛型化为 `B: burn::tensor::backend::Backend`；
//! `BackendAdapter`（Wgpu）仍保留，供 loader/check/静态校准的张量构造与对照使用。

use burn::prelude::*;

use crate::config::SdtConfig;
use crate::ops::{lif_seq, maxpool2d_3x3_s2};

/// 采用 wgpu 后端（GPU 加速）；对照与数据构造阶段使用 f32
pub type BackendAdapter = burn::backend::Wgpu;
/// wgpu 后端的设备类型（张量构造与数据搬运使用）
/// burn 0.21：Device 关联类型移到 BackendTypes supertrait
pub type Dev = <BackendAdapter as burn::tensor::backend::BackendTypes>::Device;

/// 单个卷积层的融合权重：[out, in, kH, kW] 与 [out]（泛型后端）
#[derive(Clone, Debug)]
pub struct ConvLayer<B: Backend> {
    pub w: Tensor<B, 4>,
    pub b: Tensor<B, 1>,
}

/// 全部模型权重（BN 已融合，泛型后端）
#[derive(Clone, Debug)]
pub struct SdtWeights<B: Backend> {
    // patch_embed
    pub pe_proj: [ConvLayer<B>; 4],
    pub pe_rpe: ConvLayer<B>,
    // blocks
    pub blk: Vec<BlockWeights<B>>,
    // head
    pub head_w: Tensor<B, 2>,
    pub head_b: Tensor<B, 1>,
}

/// 单个 block 的权重（泛型后端）
#[derive(Clone, Debug)]
pub struct BlockWeights<B: Backend> {
    pub q: ConvLayer<B>,
    pub k: ConvLayer<B>,
    pub v: ConvLayer<B>,
    /// talking_heads：Conv1d(h, h, 1)，权重 [h, h]（去掉了长度为 1 的最后一维）。
    /// 注意：PyTorch 的 MS_SSA_Conv.forward 实际并未调用 talking_heads Conv1d，
    /// 因此该权重保留加载但前向不使用（为兼容 loader 而保留字段）。
    #[allow(dead_code)]
    pub th_w: Tensor<B, 2>,
    pub proj: ConvLayer<B>,
    pub fc1: ConvLayer<B>,
    pub fc2: ConvLayer<B>,
    /// block 序号（0 起），用于生成分层钩子的键名（如 blk0_q_lif）
    pub layer: usize,
}

/// 分层钩子：收集前向过程中各中间张量的克隆（泛型后端，用于静态校准）。
#[derive(Default)]
pub struct LayerTaps<B: Backend> {
    /// 中间张量：键名 -> [T, B, C, ...] 张量（键名与 PyTorch 导出键一致）
    pub map: std::collections::HashMap<String, Tensor<B, 5>>,
}

impl<B: Backend> LayerTaps<B> {
    /// 记录一个中间张量（克隆进 map）
    fn tap(&mut self, key: &str, t: &Tensor<B, 5>) {
        self.map.insert(key.to_string(), t.clone());
    }
}

/// 便捷构造 4 维张量
fn t4(data: ndarray::ArrayD<f32>, device: &Dev) -> Tensor<BackendAdapter, 4> {
    let s = data.shape().to_vec();
    let flat = data.into_raw_vec();
    Tensor::<BackendAdapter, 1>::from_floats(flat.as_slice(), device)
        .reshape([s[0], s[1], s[2], s[3]])
}

/// 便捷构造 1 维张量
fn t1(data: ndarray::ArrayD<f32>, device: &Dev) -> Tensor<BackendAdapter, 1> {
    let flat = data.into_raw_vec();
    Tensor::<BackendAdapter, 1>::from_floats(flat.as_slice(), device)
}

/// 便捷构造 2 维张量
fn t2(data: ndarray::ArrayD<f32>, device: &Dev) -> Tensor<BackendAdapter, 2> {
    let s = data.shape().to_vec();
    let flat = data.into_raw_vec();
    Tensor::<BackendAdapter, 1>::from_floats(flat.as_slice(), device).reshape([s[0], s[1]])
}

/// 从 NPZ 读出的数组字典构建权重集合
pub fn load_weights(
    npz: &std::collections::HashMap<String, ndarray::ArrayD<f32>>,
    cfg: &SdtConfig,
    device: &Dev,
) -> SdtWeights<BackendAdapter> {
    let get = |k: &str| -> ndarray::ArrayD<f32> {
        npz.get(k)
            .unwrap_or_else(|| panic!("NPZ 中缺少键: {}", k))
            .clone()
    };

    let mut pe_proj = Vec::new();
    for i in 0..4 {
        pe_proj.push(ConvLayer {
            w: t4(get(&format!("pe_proj{}_w", i)), device),
            b: t1(get(&format!("pe_proj{}_b", i)), device),
        });
    }
    let pe_rpe = ConvLayer {
        w: t4(get("pe_rpe_w"), device),
        b: t1(get("pe_rpe_b"), device),
    };

    let mut blk = Vec::new();
    for j in 0..cfg.depths {
        // talking_heads 权重 [h, h, 1] -> [h, h]
        let th = get(&format!("blk{}_th_w", j));
        let ts = th.shape().to_vec();
        let th_flat = th.into_raw_vec();
        let th_w = Tensor::<BackendAdapter, 1>::from_floats(th_flat.as_slice(), device)
            .reshape([ts[0], ts[1]]);
        blk.push(BlockWeights {
            q: ConvLayer { w: t4(get(&format!("blk{}_q_w", j)), device), b: t1(get(&format!("blk{}_q_b", j)), device) },
            k: ConvLayer { w: t4(get(&format!("blk{}_k_w", j)), device), b: t1(get(&format!("blk{}_k_b", j)), device) },
            v: ConvLayer { w: t4(get(&format!("blk{}_v_w", j)), device), b: t1(get(&format!("blk{}_v_b", j)), device) },
            th_w,
            proj: ConvLayer { w: t4(get(&format!("blk{}_proj_w", j)), device), b: t1(get(&format!("blk{}_proj_b", j)), device) },
            fc1: ConvLayer { w: t4(get(&format!("blk{}_fc1_w", j)), device), b: t1(get(&format!("blk{}_fc1_b", j)), device) },
            fc2: ConvLayer { w: t4(get(&format!("blk{}_fc2_w", j)), device), b: t1(get(&format!("blk{}_fc2_b", j)), device) },
            layer: j,
        });
    }

    SdtWeights {
        pe_proj: pe_proj.try_into().expect("pe_proj 长度必须为 4"),
        pe_rpe,
        blk,
        head_w: t2(get("head_w"), device),
        head_b: t1(get("head_b"), device),
    }
}

/// 2D 卷积（stride=1，padding 按 kernel 实际尺寸自动取 (k-1)/2：3x3→1，1x1→0）。
/// 注意：PyTorch 侧 q/k/v/proj/fc1/fc2 是 1x1 卷积（padding=0），
/// SPS 各级是 3x3 卷积（padding=1），必须区分，否则 1x1 卷积会把空间尺寸撑大。
fn conv2d_auto<B: Backend>(x: Tensor<B, 4>, layer: &ConvLayer<B>) -> Tensor<B, 4> {
    let kd = layer.w.dims();
    let k = kd[2]; // 正方形 kernel 尺寸
    let pad = (k - 1) / 2;
    burn::tensor::module::conv2d(
        x,
        layer.w.clone(),
        Some(layer.b.clone()),
        burn::tensor::ops::ConvOptions::new([1, 1], [pad, pad], [1, 1], 1),
    )
}

/// LIF 序列（时间维在 dim 0）：[T, B, C, H, W] 输入 -> 同形状脉冲
fn lif<B: Backend>(x: Tensor<B, 5>, threshold: f64) -> Tensor<B, 5> {
    lif_seq(x, threshold)
}

/// [T, B, C, H, W] -> flatten 时间+batch -> conv -> 还原
fn conv_merge_tb<B: Backend>(x: Tensor<B, 5>, layer: &ConvLayer<B>) -> Tensor<B, 5> {
    let d = x.dims();
    let merged = x.reshape([d[0] * d[1], d[2], d[3], d[4]]);
    let y = conv2d_auto(merged, layer);
    let yd = y.dims();
    y.reshape([d[0], d[1], yd[1], yd[2], yd[3]])
}

/// [T, B, C, H, W] 上按 (T,B) 合并做 3x3/2 maxpool，再还原
fn pool_tb<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    let d = x.dims();
    let merged = x.reshape([d[0] * d[1], d[2], d[3], d[4]]);
    let y = maxpool2d_3x3_s2(merged);
    let yd = y.dims();
    y.reshape([d[0], d[1], yd[1], yd[2], yd[3]])
}

/// SPS 前向（pooling_stat="0011"）
/// 输入 [T, B, C_in, H_img, W_img] -> 输出 [T, B, embed, H_img/4, W_img/4]
/// 与 PyTorch module/sps.py 严格一致：
/// - 第 0/1 级：conv -> LIF（不池化）
/// - 第 2 级：conv -> LIF -> maxpool（LIF 在池化之前）
/// - 第 3 级：conv -> maxpool -> LIF（x_feat 为池化后、LIF 前的特征）
pub fn forward_sps<B: Backend>(
    x: Tensor<B, 5>,
    w: &SdtWeights<B>,
    cfg: &SdtConfig,
) -> Tensor<B, 5> {
    // 阶段 0：conv -> LIF（无池化）
    let mut xf = lif(conv_merge_tb(x, &w.pe_proj[0]), 1.0);

    // 阶段 1：conv -> LIF（无池化）
    xf = lif(conv_merge_tb(xf, &w.pe_proj[1]), 1.0);

    // 阶段 2：conv -> LIF -> pool（与 PyTorch 一致：LIF 在 maxpool2 之前）
    xf = lif(conv_merge_tb(xf, &w.pe_proj[2]), 1.0);
    xf = pool_tb(xf);

    // 阶段 3：conv -> pool -> LIF，保留 pool 后特征用于 rpe 残差
    let xf3_pool = pool_tb(conv_merge_tb(xf, &w.pe_proj[3]));
    let x_feat = xf3_pool.clone();
    let xf3 = lif(x_feat.clone(), 1.0);

    // rpe：Conv3x3(feat) + x_feat（残差加的是 LIF 前的 x_feat）
    let rpe = conv_merge_tb(xf3, &w.pe_rpe);
    let out = rpe + x_feat;
    let _ = cfg; // 预留：当前 SPS 不需要额外配置
    out
}

/// [T, B, C, N] 视角变形：输入 [T,B,C,H,W] -> [T, B, heads, N, head_dim]
fn reshape_heads<B: Backend>(
    x: Tensor<B, 5>,
    heads: usize,
    head_dim: usize,
) -> Tensor<B, 5> {
    // 等价于 PyTorch: q.flatten(3).transpose(-1,-2).reshape(T,B,N,heads,hd).permute(0,1,3,2,4)
    // Burn 的 reshape 在非连续张量上会自动物化（materialize），因此直接写即可。
    let d = x.dims();
    let (t, b, _c) = (d[0], d[1], d[2]);
    x.reshape([t, b, heads * head_dim, d[3] * d[4]])
        .swap_dims(2, 3)
        .reshape([t, b, d[3] * d[4], heads, head_dim])
        .permute([0, 1, 3, 2, 4])
}

/// [T, B, heads, N, hd] -> [T, B, C, H, W]
fn unshape_heads<B: Backend>(
    x: Tensor<B, 5>,
    heads: usize,
    head_dim: usize,
    hh: usize,
    ww: usize,
) -> Tensor<B, 5> {
    // [T,B,heads,N,hd] -> [T,B,N,heads,hd] -> [T,B,N,C] -> [T,B,C,N] -> [T,B,C,H,W]
    let d = x.dims();
    let (t, b, n) = (d[0], d[1], d[3]);
    let c = heads * head_dim;
    x.permute([0, 1, 3, 2, 4])
        .reshape([t, b, n, c])
        .swap_dims(2, 3)
        .reshape([t, b, c, hh, ww])
}

/// SSA 前向：线性脉冲注意力（direct_xor）
/// 输入 [T, B, C, H, W]，输出同形状
/// 与 PyTorch MS_SSA_Conv.forward 严格一致：kv_sum 直接过 LIF(0.5)，
/// 不做 talking_heads Conv1d 头间混合。
pub fn forward_ssa<B: Backend>(
    x: Tensor<B, 5>,
    bw: &BlockWeights<B>,
    cfg: &SdtConfig,
) -> Tensor<B, 5> {
    let d = x.dims();
    let (_t, _b, c, hh, ww) = (d[0], d[1], d[2], d[3], d[4]);
    let heads = cfg.num_heads;
    let head_dim = c / heads;

    // shortcut LIF
    let xs = lif(x.clone(), 1.0);
    let identity = x; // 残差取 LIF 之前的输入（与 PyTorch 一致）

    // q/k/v conv
    let xq = conv_merge_tb(xs.clone(), &bw.q);
    let xk = conv_merge_tb(xs.clone(), &bw.k);
    let xv = conv_merge_tb(xs, &bw.v);

    // q/k/v LIF
    let q = lif(xq, 1.0);
    let k = lif(xk, 1.0);
    let v = lif(xv, 1.0);

    // 变形为 [T, B, heads, N, head_dim]
    let qh = reshape_heads(q, heads, head_dim);
    let kh = reshape_heads(k, heads, head_dim);
    let vh = reshape_heads(v, heads, head_dim);

    // kv = k ⊙ v，按 token 求和 => [T, B, heads, 1, head_dim]
    let kv = (kh * vh.clone()).sum_dim(3);
    #[cfg(feature = "debug_shape")]
    eprintln!("[调试] kv dims = {:?}, qh dims = {:?}", kv.dims(), qh.dims());
    // kv_sum 直接过 LIF(0.5)（PyTorch 只调用 talking_heads_lif，不做 Conv1d 混合）
    let kv_spike = lif(kv, 0.5);

    // x = q ⊙ kv（广播相乘）
    let xattn = qh * kv_spike;

    // 还原为 [T, B, C, H, W]（proj conv 的输入，与 PyTorch blk{j}_x_attn 对应）
    let xo = unshape_heads(xattn, heads, head_dim, hh, ww);

    // proj conv + 残差
    let xp = conv_merge_tb(xo, &bw.proj);
    let out = xp + identity;
    out
}

/// MLP 前向
pub fn forward_mlp<B: Backend>(
    x: Tensor<B, 5>,
    bw: &BlockWeights<B>,
    _cfg: &SdtConfig,
) -> Tensor<B, 5> {
    let identity = x.clone();
    // fc1：LIF -> conv（无残差：hidden = dim*mlp_ratio != dim）
    let x1 = lif(x, 1.0);
    let x1c = conv_merge_tb(x1, &bw.fc1);
    // fc2：LIF -> conv，然后 x + identity
    let x2 = lif(x1c, 1.0);
    let x2c = conv_merge_tb(x2, &bw.fc2);
    let out = x2c + identity;
    out
}

/// 完整模型前向：返回 [B, num_classes] logits（时间平均后）。
/// 训练与对照共用本实现：权重只读（Tensor 克隆），不收集分层钩子。
pub fn forward_full<B: Backend>(
    x_in: Tensor<B, 5>,
    w: &SdtWeights<B>,
    cfg: &SdtConfig,
) -> Tensor<B, 2> {
    let d = x_in.dims();
    let (t, b) = (d[0], d[1]);

    // patch_embed（SPS）
    let mut x = forward_sps(x_in, w, cfg);
    #[cfg(feature = "debug_shape")]
    eprintln!("[调试] SPS 输出 dims = {:?}", x.dims());

    // blocks
    for bw in &w.blk {
        #[cfg(feature = "debug_shape")]
        eprintln!("[调试] block 输入 dims = {:?}", x.dims());
        let attn_out = forward_ssa(x, bw, cfg);
        x = forward_mlp(attn_out, bw, cfg);
    }

    // flatten(3).mean(3)：[T,B,C,H,W] -> [T,B,C]
    let fd = x.dims();
    let feat = x
        .reshape([fd[0], fd[1], fd[2], fd[3] * fd[4]])
        .sum_dim(3)
        .div_scalar((fd[3] * fd[4]) as f32);

    // head_lif（阈值 1.0）：feat 为 [T,B,C]（3 维），用通用 lif_seq 处理
    let feat_spike = lif_seq(feat, 1.0);

    // head linear：[T*B,C] x [C,num_classes] -> [T*B,num_classes]
    let logits_tb = feat_spike
        .reshape([t * b, fd[2]])
        .matmul(w.head_w.clone().swap_dims(0, 1))
        + w.head_b.clone().unsqueeze::<2>().unsqueeze::<3>().reshape([1, cfg.num_classes]);

    // 时间平均：[T,B,num_classes] -> [B,num_classes]
    // burn 0.21：squeeze 不再带索引参数（形状由泛型 D2 唯一确定）
    let ld = logits_tb.dims();
    logits_tb
        .reshape([t, b, ld[1]])
        .sum_dim(0)
        .squeeze::<2>()
        .div_scalar(t as f32)
}

/// 带 taps 的完整前向（静态校准专用）：按 PyTorch `calibrate_fused_weights` 的
/// 层清单，收集每个「输出直接喂给 LIF 的层」的输出：
/// - pe_proj0..3：SPS 各级 conv 输出（LIF 输入）
/// - pe_rpe：rpe conv 输出（后接残差相加后进入首个 block 的 shortcut LIF，即 LIF 输入）
/// - blk{j}_q/k/v：q/k/v conv 输出（LIF 输入）
/// - blk{j}_proj：proj conv 输出（残差相加后进入 MLP 的 fc1_lif，即 LIF 输入）
/// - blk{j}_fc1/fc2：fc1/fc2 conv 输出（fc2 输出残差相加后进入下一 block/head_lif）
/// - head：Linear 输出 [T, B, num_classes]（返回前 squeeze 成 [T, B, C, 1, 1] 以统一 5 维）
/// 键名与 PyTorch 脚本 layer_list 完全一致（head 输出统计在 [N, num_classes] 上）。
pub fn forward_taps<B: Backend>(
    x_in: Tensor<B, 5>,
    w: &SdtWeights<B>,
    cfg: &SdtConfig,
) -> LayerTaps<B> {
    let mut taps = LayerTaps::<B>::default();

    // ---- SPS 阶段 ----
    // 阶段 0：conv -> LIF（taps 记录 conv 输出，即 lif0 的输入）
    let c0 = conv_merge_tb(x_in, &w.pe_proj[0]);
    taps.tap("pe_proj0", &c0);
    let mut xf = lif(c0, 1.0);

    // 阶段 1：conv -> LIF
    let c1 = conv_merge_tb(xf, &w.pe_proj[1]);
    taps.tap("pe_proj1", &c1);
    xf = lif(c1, 1.0);

    // 阶段 2：conv -> LIF -> pool
    let c2 = conv_merge_tb(xf, &w.pe_proj[2]);
    taps.tap("pe_proj2", &c2);
    xf = pool_tb(lif(c2, 1.0));

    // 阶段 3：conv -> pool -> LIF
    // taps 记录 conv3 输出（pool/LIF 之前）——与 PyTorch 钩子挂在 proj_conv3 模块上的口径一致
    let c3 = conv_merge_tb(xf, &w.pe_proj[3]);
    taps.tap("pe_proj3", &c3);
    let xf3_pool = pool_tb(c3);
    let x_feat = xf3_pool.clone();
    // rpe conv 输出（与 x_feat 残差相加后作为 SPS 输出进入 block 的 shortcut_lif）
    let rpe = conv_merge_tb(lif(x_feat.clone(), 1.0), &w.pe_rpe);
    taps.tap("pe_rpe", &rpe);
    let mut x = rpe + x_feat;

    // ---- blocks 阶段 ----
    for bw in &w.blk {
        let d = x.dims();
        let (_t, _b, c, hh, ww) = (d[0], d[1], d[2], d[3], d[4]);
        let heads = cfg.num_heads;
        let head_dim = c / heads;
        let j = bw.layer;

        // shortcut LIF -> q/k/v 三分支（taps 记录各 conv 输出）
        let xs = lif(x.clone(), 1.0);
        let identity = x;
        let xq = conv_merge_tb(xs.clone(), &bw.q);
        taps.tap(&format!("blk{}_q", j), &xq);
        let xk = conv_merge_tb(xs.clone(), &bw.k);
        taps.tap(&format!("blk{}_k", j), &xk);
        let xv = conv_merge_tb(xs, &bw.v);
        taps.tap(&format!("blk{}_v", j), &xv);

        let q = lif(xq, 1.0);
        let k = lif(xk, 1.0);
        let v = lif(xv, 1.0);

        let qh = reshape_heads(q, heads, head_dim);
        let kh = reshape_heads(k, heads, head_dim);
        let vh = reshape_heads(v, heads, head_dim);

        let kv = (kh * vh.clone()).sum_dim(3);
        let kv_spike = lif(kv, 0.5);
        let xattn = qh * kv_spike;
        let xo = unshape_heads(xattn, heads, head_dim, hh, ww);

        // proj conv 输出（残差相加后进入 MLP 的 fc1_lif）
        let xp = conv_merge_tb(xo, &bw.proj);
        taps.tap(&format!("blk{}_proj", j), &xp);
        x = xp + identity;

        // MLP：fc1_lif -> conv -> fc2_lif -> conv -> 残差
        let identity = x.clone();
        let x1 = lif(x, 1.0);
        let x1c = conv_merge_tb(x1, &bw.fc1);
        taps.tap(&format!("blk{}_fc1", j), &x1c);
        let x2 = lif(x1c, 1.0);
        let x2c = conv_merge_tb(x2, &bw.fc2);
        taps.tap(&format!("blk{}_fc2", j), &x2c);
        x = x2c + identity;
    }

    // ---- head 阶段 ----
    // flatten(3).mean(3) -> head_lif -> Linear
    let fd = x.dims();
    let (t, b) = (fd[0], fd[1]);
    let feat = x
        .reshape([t, b, fd[2], fd[3] * fd[4]])
        .sum_dim(3)
        .div_scalar((fd[3] * fd[4]) as f32);
    let feat_spike = lif_seq(feat, 1.0);
    let logits_tb = feat_spike
        .reshape([t * b, fd[2]])
        .matmul(w.head_w.clone().swap_dims(0, 1))
        + w.head_b.clone().unsqueeze::<2>().unsqueeze::<3>().reshape([1, cfg.num_classes]);
    // head 输出 [T*B, num_classes] -> 还原 [T, B, num_classes, 1, 1]（统一 5 维存放）
    let ld = logits_tb.dims();
    let head5 = logits_tb.reshape([t, b, ld[1], 1, 1]);
    taps.tap("head", &head5);

    taps
}
