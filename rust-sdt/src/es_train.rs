//! EggRoll 演化策略（ES）训练：无反向传播训练 SDT 模型。
//!
//! 算法来源：HyperscaleES（eggroll）及其 Rust 迁移
//! （`F:\PythonProject\HyperscaleES\burn_impl\hyperscalees-noiser\src\eggroll.rs`），
//! 本模块把其核心语义对齐移植到 SDT（卷积型 SNN-Transformer）上：
//!
//! - **LoRA 低秩扰动**（eggroll 核心）：把每个卷积权重 [O,I,k,k] 展平为 2D [O, I·k·k]，
//!   每个候选 c 的扰动 ΔW_c = ±(σ/√r)·A_c·B_cᵀ（A_c∈R^{a×r}, B_c∈R^{b×r}，标准正态）。
//!   候选 2k 与 2k+1 为反对称对（共享同一随机数、符号相反）。
//! - **偏置 dense 扰动**：N(0,1)·(±σ)（eggroll 的 nonlora 路径）。
//! - **种子噪声**：与迁移版一致，xorshift64 + Box-Muller，`noise_seed(key, true_epoch, thread)`
//!   派生（对应 jax fold_in）；`noise_reuse=0` 时噪声只依赖 (key, thread)，跨 epoch 固定，
//!   因此启动时一次性预生成全部候选的 ΔW 缓存（VRAM ≈ pop × 参数量 × 4B）。
//! - **fitness**：loglik = log_softmax(logits)[label]（迁移版重测结论：新配置下 loglik
//!   优于硬 0/1 与 margin 类，见 HyperscaleES docs/snn_es_mnist_experiment.md §7.5 重测）。
//!   候选 fitness = 其图像块上的 loglik 之和。
//! - **更新**：全局 z-score 仿射修正（与迁移版 `combine_affine_grads` 严格一致）：
//!   `grads = -(Σ_c raw_c·ΔW_c - mean·Σ_c ΔW_c) / (std·√pop)`，随后一次 AdamW 步
//!   （β1=0.9, β2=0.999, eps=1e-8, decoupled wd=1e-4，与迁移版 Solver::AdamW 一致）。
//! - **SDT 适配**：eggroll 原版是逐样本噪声（vmap 矩阵乘法）。SDT 全部是卷积权重，
//!   逐样本噪声需逐样本换核（无法批量），故把「候选」从单张图放大为一个 es_batch
//!   图像块（块内共享同一扰动），结构上与逐样本方案同构（候选 c ↔ 样本 c，只是每个
//!   候选的 fitness 从单样本 loglik 变为 m 张图的 loglik 之和）。
//! - **代（generation）与 epoch**：一个 epoch = 全部 8000 张训练图过一遍 =
//!   `8000/(pop·es_batch)` 个 generation（不能整除的尾部丢弃），每个 generation 一次
//!   AdamW 更新（对应迁移版「每 epoch 一代一更新」，但更新次数多 pop/es_batch 倍）。
//! - **后端**：全程运行在内层 wgpu 后端（`BackendAdapter`，无 autodiff 图），从根上
//!   规避「autodiff 前向图节点无 backward 消费时钉死显存」的泄漏（见 OOM_POSTMORTEM §②）；
//!   每 epoch 末 sync + memory_cleanup（顺序约束与 train.rs 相同）。

use burn::prelude::*;

use crate::config::SdtConfig;
use crate::loader::{shuffled_indices, Cifar10Npz, Split};
use crate::model::{
    forward_full, forward_sps, reshape_heads, unshape_heads, BackendAdapter, BlockWeights,
    QamWeights,
    ConvLayer, Dev, SdtWeights,
};
use crate::ops::{lif_seq, lif_seq_bth, lif_seq_relaxed};
use crate::train::{to_autodiff_weights, to_wgpu_weights, AutodiffBackend, AutodiffDevice, TrainSdt};

/// 本模块全部张量所在后端（内层 wgpu，无 autodiff）
type B = BackendAdapter;

/// 黄金比例乘子（与迁移版 KEY_MUL 一致，用于派生每参数 base_key）
const KEY_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

// ---------------------------------------------------------------------------
// 确定性噪声（与迁移版 DeterministicNoise / noise_seed 逐位一致的移植）
// ---------------------------------------------------------------------------

/// 种子化标准正态采样器：xorshift64 + Box-Muller（与迁移版实现相同，
/// 并使用 cos/sin 双输出——每次 Box-Muller 产出两个正态，吞吐 ×2）
struct DeterministicNoise {
    state: u64,
    pending: Option<f32>,
}

impl DeterministicNoise {
    fn new(seed: u64) -> Self {
        Self {
            state: seed | 1,
            pending: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// 单个 N(0,1)（Box-Muller，sin 支路缓存到下次调用）
    fn standard_normal(&mut self) -> f32 {
        if let Some(v) = self.pending.take() {
            return v;
        }
        let u1 = self.next_unit().max(1e-9);
        let u2 = self.next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        self.pending = Some(radius * theta.sin());
        radius * theta.cos()
    }
}

/// 从 (per-param key, true_epoch, true_thread) 派生噪声种子
/// （对应迁移版 `noise_seed`，即 jax `fold_in(fold_in(key, true_epoch), true_thread)`）
fn noise_seed(key: u64, true_epoch: i32, true_thread: i32) -> u64 {
    let mut h = key ^ 0x9E37_79B9_7F4A_7C15u64;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= (true_epoch as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h = h.rotate_left(17) ^ (true_thread as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 32;
    h
}

// ---------------------------------------------------------------------------
// ES 参数槽：把 SdtWeights 线性化为 (2D 展平, 形状, LoRA/dense) 列表
// ---------------------------------------------------------------------------

/// 槽位的原始形状（写回/组装时恢复）
#[derive(Clone, Debug)]
enum OrigShape {
    /// 卷积核 [O, I, kH, kW]
    W4([usize; 4]),
    /// 2D 权重（head Linear [num_classes, C]）
    W2([usize; 2]),
    /// 偏置 [O]
    B1(usize),
}

impl OrigShape {
    /// 展平后的 2D 形状 [a, b]（a=第一维，b=其余元素数）
    fn flat2d(&self) -> [usize; 2] {
        match self {
            OrigShape::W4(s) => [s[0], s[1] * s[2] * s[3]],
            OrigShape::W2(s) => [s[0], s[1]],
            OrigShape::B1(n) => [*n, 1],
        }
    }

    fn orig_dims(&self) -> Vec<usize> {
        match self {
            OrigShape::W4(s) => s.to_vec(),
            OrigShape::W2(s) => s.to_vec(),
            OrigShape::B1(n) => vec![*n],
        }
    }
}

/// ES 参数清单。槽位顺序（slots_of / flatten_params / assemble / perturb_weights
/// 四处必须严格一致）：pe_proj0..3(w,b) → pe_rpe(w,b) → 每 block q/k/v/proj/fc1/fc2(w,b)
/// → head(w,b)。th_w 不参与 ES（前向不使用），仅随组装原样带过。
struct EsSlots {
    /// 每槽原始形状
    shapes: Vec<OrigShape>,
    /// 是否 LoRA 槽（权重=true 用低秩扰动；偏置=false 用 dense 噪声）
    lora: Vec<bool>,
    /// 每槽确定性 base_key
    keys: Vec<u64>,
    /// 参数总量（不含 th_w）
    num_params: usize,
    /// th_w 引用（组装时克隆）
    th_w: Vec<Tensor<B, 2>>,
}

/// 按槽形状把任意秩张量展平为 2D [a,b]
fn flatten_slot<B: burn::tensor::backend::Backend, const D: usize>(
    shape: &OrigShape,
    t: &Tensor<B, D>,
) -> Tensor<B, 2> {
    let [a, b] = shape.flat2d();
    t.clone().reshape([a, b])
}

/// 从权重集合提取 ES 槽
fn slots_of(w: &SdtWeights<B>, depths: usize) -> EsSlots {
    let mut shapes: Vec<OrigShape> = Vec::new();
    let mut lora: Vec<bool> = Vec::new();

    for i in 0..4 {
        shapes.push(OrigShape::W4(w.pe_proj[i].w.dims()));
        lora.push(true);
        shapes.push(OrigShape::B1(w.pe_proj[i].b.dims()[0]));
        lora.push(false);
    }
    shapes.push(OrigShape::W4(w.pe_rpe.w.dims()));
    lora.push(true);
    shapes.push(OrigShape::B1(w.pe_rpe.b.dims()[0]));
    lora.push(false);
    for j in 0..depths {
        let b = &w.blk[j];
        for conv in [&b.q, &b.k, &b.v, &b.proj, &b.fc1, &b.fc2] {
            shapes.push(OrigShape::W4(conv.w.dims()));
            lora.push(true);
            shapes.push(OrigShape::B1(conv.b.dims()[0]));
            lora.push(false);
        }
    }
    shapes.push(OrigShape::W2(w.head_w.dims()));
    lora.push(true);
    shapes.push(OrigShape::B1(w.head_b.dims()[0]));
    lora.push(false);

    let num_params: usize = shapes
        .iter()
        .map(|s| s.orig_dims().iter().product::<usize>())
        .sum();
    let keys: Vec<u64> = (0..shapes.len())
        .map(|i| (i as u64 + 1).wrapping_mul(KEY_MUL))
        .collect();
    let th_w = w.blk.iter().map(|b| b.th_w.clone()).collect();
    EsSlots {
        shapes,
        lora,
        keys,
        num_params,
        th_w,
    }
}

/// 取主参数的 2D 展平视图（与 slots_of 槽顺序一致）
fn flatten_params(slots: &EsSlots, w: &SdtWeights<B>, depths: usize) -> Vec<Tensor<B, 2>> {
    let mut out: Vec<Tensor<B, 2>> = Vec::new();
    let mut si = 0usize;
    for i in 0..4 {
        // 泛型秩差异导致无法用统一闭包，逐槽显式展开：
        out.push(flatten_slot(&slots.shapes[si], &w.pe_proj[i].w));
        si += 1;
        out.push(flatten_slot(&slots.shapes[si], &w.pe_proj[i].b));
        si += 1;
    }
    out.push(flatten_slot(&slots.shapes[si], &w.pe_rpe.w));
    si += 1;
    out.push(flatten_slot(&slots.shapes[si], &w.pe_rpe.b));
    si += 1;
    for j in 0..depths {
        let b = &w.blk[j];
        for conv in [&b.q, &b.k, &b.v, &b.proj, &b.fc1, &b.fc2] {
            out.push(flatten_slot(&slots.shapes[si], &conv.w));
            si += 1;
            out.push(flatten_slot(&slots.shapes[si], &conv.b));
            si += 1;
        }
    }
    out.push(flatten_slot(&slots.shapes[si], &w.head_w));
    si += 1;
    out.push(flatten_slot(&slots.shapes[si], &w.head_b));
    out
}

/// 把 2D 主参数列表组装回 SdtWeights（与 slots_of 槽顺序严格一致）
fn assemble(slots: &EsSlots, depths: usize, flat: &[Tensor<B, 2>]) -> SdtWeights<B> {
    let mut idx = 0usize;
    let mut take4 = |flat: &[Tensor<B, 2>], slots: &EsSlots, idx: &mut usize| -> Tensor<B, 4> {
        let d = slots.shapes[*idx].orig_dims();
        *idx += 1;
        flat[*idx - 1].clone().reshape([d[0], d[1], d[2], d[3]])
    };
    let mut take1 = |flat: &[Tensor<B, 2>], slots: &EsSlots, idx: &mut usize| -> Tensor<B, 1> {
        let n = slots.shapes[*idx].orig_dims()[0];
        *idx += 1;
        flat[*idx - 1].clone().reshape([n])
    };

    let mut pe_proj = Vec::new();
    for _ in 0..4 {
        pe_proj.push(ConvLayer {
            w: take4(flat, slots, &mut idx),
            b: take1(flat, slots, &mut idx),
        });
    }
    let pe_rpe = ConvLayer {
        w: take4(flat, slots, &mut idx),
        b: take1(flat, slots, &mut idx),
    };
    let mut blk = Vec::new();
    for j in 0..depths {
        let q = ConvLayer {
            w: take4(flat, slots, &mut idx),
            b: take1(flat, slots, &mut idx),
        };
        let k = ConvLayer {
            w: take4(flat, slots, &mut idx),
            b: take1(flat, slots, &mut idx),
        };
        let v = ConvLayer {
            w: take4(flat, slots, &mut idx),
            b: take1(flat, slots, &mut idx),
        };
        let proj = ConvLayer {
            w: take4(flat, slots, &mut idx),
            b: take1(flat, slots, &mut idx),
        };
        let fc1 = ConvLayer {
            w: take4(flat, slots, &mut idx),
            b: take1(flat, slots, &mut idx),
        };
        let fc2 = ConvLayer {
            w: take4(flat, slots, &mut idx),
            b: take1(flat, slots, &mut idx),
        };
        blk.push(BlockWeights {
            q,
            k,
            v,
            th_w: slots.th_w[j].clone(),
            proj,
            fc1,
            fc2,
            layer: j,
        });
    }
    let head_dims = slots.shapes[idx].flat2d();
    let head_w = flat[idx].clone().reshape([head_dims[0], head_dims[1]]);
    idx += 1;
    let n = slots.shapes[idx].orig_dims()[0];
    let head_b = flat[idx].clone().reshape([n]);

    SdtWeights {
        pe_proj: pe_proj.try_into().expect("pe_proj 长度必须为 4"),
        pe_rpe,
        blk,
        head_w,
        head_b,
    }
}

/// 组装「基础权重 + 单个候选扰动」的权重集合（槽顺序同 slots_of）
fn perturb_weights(
    slots: &EsSlots,
    depths: usize,
    base: &SdtWeights<B>,
    delta: &[Tensor<B, 2>],
) -> SdtWeights<B> {
    // 显式逐槽相加（w_pert = base + ΔW）
    let mut k = 0usize;
    let add4x = |w: &Tensor<B, 4>,
                     d: &Tensor<B, 2>,
                     slots: &EsSlots,
                     k: &mut usize|
     -> Tensor<B, 4> {
        let dims = slots.shapes[*k].orig_dims();
        *k += 1;
        w.clone() + d.clone().reshape([dims[0], dims[1], dims[2], dims[3]])
    };
    let add1x = |b: &Tensor<B, 1>,
                     d: &Tensor<B, 2>,
                     slots: &EsSlots,
                     k: &mut usize|
     -> Tensor<B, 1> {
        let n = slots.shapes[*k].orig_dims()[0];
        *k += 1;
        b.clone() + d.clone().reshape([n])
    };

    let mut pe_proj = Vec::new();
    for i in 0..4 {
        pe_proj.push(ConvLayer {
            w: add4x(&base.pe_proj[i].w, &delta[k], slots, &mut k),
            b: add1x(&base.pe_proj[i].b, &delta[k], slots, &mut k),
        });
    }
    let pe_rpe = ConvLayer {
        w: add4x(&base.pe_rpe.w, &delta[k], slots, &mut k),
        b: add1x(&base.pe_rpe.b, &delta[k], slots, &mut k),
    };
    let mut blk = Vec::new();
    for j in 0..depths {
        let b = &base.blk[j];
        let q = ConvLayer {
            w: add4x(&b.q.w, &delta[k], slots, &mut k),
            b: add1x(&b.q.b, &delta[k], slots, &mut k),
        };
        let kk = ConvLayer {
            w: add4x(&b.k.w, &delta[k], slots, &mut k),
            b: add1x(&b.k.b, &delta[k], slots, &mut k),
        };
        let v = ConvLayer {
            w: add4x(&b.v.w, &delta[k], slots, &mut k),
            b: add1x(&b.v.b, &delta[k], slots, &mut k),
        };
        let proj = ConvLayer {
            w: add4x(&b.proj.w, &delta[k], slots, &mut k),
            b: add1x(&b.proj.b, &delta[k], slots, &mut k),
        };
        let fc1 = ConvLayer {
            w: add4x(&b.fc1.w, &delta[k], slots, &mut k),
            b: add1x(&b.fc1.b, &delta[k], slots, &mut k),
        };
        let fc2 = ConvLayer {
            w: add4x(&b.fc2.w, &delta[k], slots, &mut k),
            b: add1x(&b.fc2.b, &delta[k], slots, &mut k),
        };
        blk.push(BlockWeights {
            q,
            k: kk,
            v,
            th_w: slots.th_w[j].clone(),
            proj,
            fc1,
            fc2,
            layer: j,
        });
    }
    let hd = slots.shapes[k].flat2d();
    let head_w = base.head_w.clone() + delta[k].clone().reshape([hd[0], hd[1]]);
    k += 1;
    let n = slots.shapes[k].orig_dims()[0];
    let head_b = base.head_b.clone() + delta[k].clone().reshape([n]);

    SdtWeights {
        pe_proj: pe_proj.try_into().expect("pe_proj 长度必须为 4"),
        pe_rpe,
        blk,
        head_w,
        head_b,
    }
}

// ---------------------------------------------------------------------------
// 扰动缓存预生成（noise_reuse=0：噪声跨 epoch 固定）
// ---------------------------------------------------------------------------

/// 为每个候选预生成全部槽位的扰动（2D 展平形态），返回 deltas[candidate][slot]。
///
/// **逐槽相对幅度**：SDT 校准后各层权重 std 差异极大（实测 0.057~1.4，BN 融合所致），
/// 绝对 σ 会摧毁小尺度层（pe_rpe std=0.057 时 σ=0.2 相当于 348% 扰动，logits 爆到 ±77、
/// fitness 信号全灭——冒烟实测）。因此每槽缩放取 `σ_slot = σ·std(W_slot)`：
/// - LoRA 槽：ΔW = (σ_slot/√r)·A·Bᵀ，元素 std = σ_slot（A/B 标准正态，
///   A 乘 sign·(σ_slot/√r)，B 取 lora 前 b 行、A 取后 a 行，与迁移版
///   get_lora_update_params 的切分一致）
/// - dense 槽：noise·(sign·σ_slot)
fn build_delta_cache(
    slots: &EsSlots,
    params: &[Tensor<B, 2>],
    pop: usize,
    sigma: f32,
    rank: usize,
    device: &Dev,
) -> Vec<Vec<Tensor<B, 2>>> {
    // 每槽相对幅度 σ_slot = σ·std(W_slot)（std 在 GPU 上一次求出）
    let mut slot_sigma = Vec::with_capacity(slots.shapes.len());
    for (si, p) in params.iter().enumerate() {
        let n = p.dims()[0] * p.dims()[1];
        let mean = p.clone().sum().into_scalar() / n as f32;
        let var = p.clone().powf_scalar(2.0).sum().into_scalar() / n as f32 - mean * mean;
        slot_sigma.push(sigma * var.max(0.0).sqrt());
    }
    println!(
        "[es] 逐槽扰动幅度 σ_slot（σ={}×std）：{:.4?}",
        sigma,
        slot_sigma.iter().map(|s| s * 1000.0).collect::<Vec<f32>>()
    );
    let mut deltas = Vec::with_capacity(pop);
    for c in 0..pop {
        let sign = if c % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
        let true_thread = (c / 2) as i32; // 反对称对共享随机数（thread 2k/2k+1 → k）
        let mut per_slot = Vec::with_capacity(slots.shapes.len());
        for (si, shape) in slots.shapes.iter().enumerate() {
            let [a, b] = shape.flat2d();
            let mut rng = DeterministicNoise::new(noise_seed(slots.keys[si], 0, true_thread));
            let d2 = if slots.lora[si] {
                // (a+b)×r 标准正态：前 b 行为 B，后 a 行为 A（乘 sign·σ_slot/√r）
                let base_sigma = slot_sigma[si] / (rank as f32).sqrt();
                let mut lora = vec![0.0_f32; (a + b) * rank];
                for v in lora.iter_mut() {
                    *v = rng.standard_normal();
                }
                let a_mat = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(
                        lora[b * rank..(a + b) * rank]
                            .iter()
                            .map(|v| v * sign * base_sigma)
                            .collect(),
                        vec![a, rank],
                    ),
                    device,
                );
                let b_mat = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(lora[..b * rank].to_vec(), vec![b, rank]),
                    device,
                );
                a_mat.matmul(b_mat.transpose())
            } else {
                let n = a * b;
                let mut noise = vec![0.0_f32; n];
                for v in noise.iter_mut() {
                    *v = rng.standard_normal() * sign * slot_sigma[si];
                }
                Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(noise, vec![a, b]),
                    device,
                )
            };
            per_slot.push(d2);
        }
        deltas.push(per_slot);
    }
    deltas
}

// ---------------------------------------------------------------------------
// AdamW（与迁移版 Solver::AdamW 语义一致：optax 默认 β/eps + decoupled wd）
// ---------------------------------------------------------------------------

struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u64,
    m: Vec<Tensor<B, 2>>,
    v: Vec<Tensor<B, 2>>,
}

impl AdamW {
    fn new(shapes: &[OrigShape], lr: f32, wd: f32, device: &Dev) -> Self {
        let m = shapes
            .iter()
            .map(|s| Tensor::<B, 2>::zeros(s.flat2d(), device))
            .collect();
        let v = shapes
            .iter()
            .map(|s| Tensor::<B, 2>::zeros(s.flat2d(), device))
            .collect();
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: wd,
            step: 0,
            m,
            v,
        }
    }

    /// 可变学习率调度（warmup+cosine）：每 epoch 更新一次
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// 一次更新：grads 为「已取负」的 ES 梯度（与迁移版 do_update 返回值同号）
    fn step(&mut self, params: &mut [Tensor<B, 2>], grads: &[Tensor<B, 2>]) {
        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        for i in 0..params.len() {
            let m_new = self.m[i].clone().mul_scalar(self.beta1)
                + grads[i].clone().mul_scalar(1.0 - self.beta1);
            let v_new = self.v[i].clone().mul_scalar(self.beta2)
                + grads[i].clone().powf_scalar(2.0).mul_scalar(1.0 - self.beta2);
            self.m[i] = m_new;
            self.v[i] = v_new;
            let m_hat = self.m[i].clone().mul_scalar(1.0 / bc1);
            let v_hat = self.v[i].clone().mul_scalar(1.0 / bc2);
            let denom = v_hat.sqrt().add_scalar(self.eps);
            let adam_term = (m_hat / denom).mul_scalar(-self.lr);
            params[i] = params[i].clone() + adam_term
                + params[i].clone().mul_scalar(self.weight_decay);
        }
    }
}

// ---------------------------------------------------------------------------
// loglik fitness
// ---------------------------------------------------------------------------

/// 数值稳定 log_softmax（dim=1）：x - logsumexp(x)
fn log_softmax2(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let m = x.clone().max_dim(1); // [m,1]
    let e = (x.clone() - m.clone()).exp();
    let lse = e.sum_dim(1).log() + m; // [m,1]
    x - lse
}

// ---------------------------------------------------------------------------
// ES 训练入口
// ---------------------------------------------------------------------------

/// ES 训练参数
#[derive(Clone, Debug)]
pub struct EsArgs {
    pub epochs: u32,
    pub data_dir: String,
    pub seed: u64,
    pub weights: Option<String>,
    pub time_steps: usize,
    pub no_calibrate: bool,
    /// 每代候选数（偶数；候选 2k/2k+1 为反对称对）
    pub pop: usize,
    /// 每候选评估的图像块大小（块内共享同一扰动）
    pub es_batch: usize,
    /// 扰动幅度 σ（相对值：每槽实际幅度 = σ × 该层权重 std，见 build_delta_cache）
    pub sigma: f32,
    /// LoRA 秩 r
    pub rank: usize,
    /// AdamW 学习率
    pub lr: f64,
    /// 每 N 个 epoch 评估一次验证集
    pub validate_every: u32,
    /// CSV 输出路径
    pub csv_out: String,
    /// 实现模式："factored"（默认，因式分解噪声前向，逐图候选，SPS 冻结）
    /// | "cache"（ΔW 物化缓存，块共享扰动）
    pub mode: String,
    /// factored 模式：每 chunk 的候选（图）数
    pub chunk: usize,
    /// TSES 温度松弛：fitness 前向用 σ(β(h−thr)) 松弛 LIF（定理 3）
    pub relax: bool,
    /// 松弛温度 β（= STE α）
    pub beta: f32,
    /// β 退火周期（每 N epoch ×2，上限 16；0 = 固定）
    pub beta_anneal_every: u32,
    /// S6 松弛作用域："all"（全部位点）| "ssa"（仅 SSA 位点，MLP/head 硬 LIF）
    pub relax_scope: String,
    /// 可变学习率：线性 warmup 的 epoch 数（0 = 恒定 lr）
    pub lr_warmup: u32,
    /// 可变学习率：余弦退火终值比例（lr_min = lr × lr_min_frac）
    pub lr_min_frac: f64,
    /// S4（opt-in）：β≥16 时训练 fitness 切硬 LIF（Run-1 实测长跑为负优化，默认关）
    pub hard_at_16: bool,
    /// QAM 调制（ES_MANIFOLD §12）："off" | "learnable"（m=(1+a)·cos(φ)，2 参/神经元）
    pub qam: String,
    /// QAM 位点集："head"（feat，默认）| "ssa"（xs/q/k/v/kv）| "all"（8 位点）
    pub qam_sites: String,
    /// ± 对共享同一样本（协议修正）：候选 2k/2k+1 评估同一图像、噪声反号——
    /// 对偶差分中数据难度项精确消去，深槽的扰动响应不再被图像难度方差淹没
    pub pair_shared: bool,
    /// 块结构消融："conv"（默认，六个 1×1 conv/块）| "none"（恒等直连，
    /// 注意力核直接作用于 SPS 特征，可训练= head+QAM）| "probe"（纯线性探针：
    /// SPS→均值池化→线性 head，无 LIF/无块，可训练= head）
    pub blocks: String,
    /// 冻结卷积权重槽（σ_lora=0，仅 "conv" 模式有效）：只训 bias+head+QAM——
    /// 检验“深槽只注噪不产信号”的稀释假设
    pub freeze_lora: bool,
    /// 逐候选零噪声锚点（控制变量，X1）：每 epoch 先无噪声评估当前 θ 在全部
    /// 训练图上的 loglik b₀(x)，候选 raw′ₙ = rawₙ − b₀(xₙ) 再 z-score——
    /// 图像难度项逐点扣除，z 尺度从数据方差塌缩为响应方差
    /// （GMD 核化均值偏移的锚点项 + EggRollBS 基线扣除的样本级推广）
    pub baseline_zero: bool,
    /// 逐槽步长自适应（X3，NES/CSA 式）：每槽漂移幅值 EMA，每 N epoch 以
    /// r=(ema_i/median)^0.5 缩放 σ_slot∈[0.25×,4×]——有信号的槽增噪、
    /// 死槽减噪，直接攻击“深槽噪声污染共享 σ_z”的消融发现
    pub sigma_adapt: bool,
    /// σ 自适应周期（epoch）
    pub sigma_adapt_every: u32,
}

/// 内层 wgpu 设备引用
fn es_dev() -> &'static Dev {
    static ONCE: std::sync::OnceLock<Dev> = std::sync::OnceLock::new();
    ONCE.get_or_init(Default::default)
}

/// 验证集 top-1（内层后端分批推理，逐批 sync + cleanup）
fn eval_es(
    w: &SdtWeights<B>,
    qam: Option<&QamWeights<B>>,
    data: &Cifar10Npz,
    batch: usize,
    t: usize,
    cfg: &SdtConfig,
) -> f64 {
    let device = es_dev();
    let n = data.n_test;
    let order: Vec<usize> = (0..n).collect();
    let mut correct = 0usize;
    for chunk in order.chunks(batch) {
        let (images, targets) = data.get_batch(Split::Test, chunk, t, device);
        let logits = forward_full(images, w, cfg, qam);
        let pred = logits.argmax(1).reshape([chunk.len()]);
        let pred_v: Vec<i64> = pred
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("读取预测失败");
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
        <B as burn::tensor::backend::Backend>::sync(device).expect("eval 逐批同步失败");
        <B as burn::tensor::backend::Backend>::memory_cleanup(device);
    }
    <B as burn::tensor::backend::Backend>::sync(device).expect("eval 同步失败");
    <B as burn::tensor::backend::Backend>::memory_cleanup(device);
    correct as f64 / n as f64 * 100.0
}

/// 架构感知验证（none/probe 模式）：与训练前向同架构、零噪声。
/// conv 模式直接走 eval_es（forward_full + QamWeights）。
fn eval_es_arch(
    w: &SdtWeights<B>,
    qam_eval: Option<&QamWeights<B>>,
    data: &Cifar10Npz,
    batch: usize,
    t: usize,
    cfg: &SdtConfig,
    blocks: &str,
) -> f64 {
    if blocks == "conv" {
        return eval_es(w, qam_eval, data, batch, t, cfg);
    }
    let device = es_dev();
    let n = data.n_test;
    let order: Vec<usize> = (0..n).collect();
    let mut correct = 0usize;
    let head_in = w.head_w.dims()[1];
    for chunk in order.chunks(batch) {
        let (images, targets) = data.get_batch(Split::Test, chunk, t, device);
        let c = chunk.len();
        // 零噪声因子：lora[0]=(0,0)、dense[0]=0（head 无噪声）+ QAM deltas=0
        let lora = vec![(
            Tensor::<B, 3>::zeros([c, 1, cfg.num_classes], device),
            Tensor::<B, 3>::zeros([c, 1, head_in], device),
        )];
        let mut dense = vec![Tensor::<B, 2>::zeros([c, cfg.num_classes], device)];
        let qam_run = qam_eval.map(|q| {
            let mut sites = Vec::new();
            for (s, a, p) in &q.sites {
                let nn = a.dims()[0];
                let za = Tensor::<B, 2>::zeros([c, nn], device);
                let zp = Tensor::<B, 2>::zeros([c, nn], device);
                dense.push(za.clone());
                dense.push(zp.clone());
                sites.push((
                    *s,
                    a.clone().reshape([nn, 1]),
                    p.clone().reshape([nn, 1]),
                    za,
                    zp,
                ));
            }
            let vth = q
                .vth
                .iter()
                .map(|(s, v)| (*s, *v, Tensor::<B, 1>::zeros([c], device)))
                .collect();
            QamRun { sites, vth }
        });
        let factors = ChunkFactors { lora, dense };
        let logits = match blocks {
            "none" => es_forward_noconv(images, w, &factors, cfg, false, false, 4.0, qam_run.as_ref()),
            "probe" => es_forward_probe(images, w, &factors, cfg),
            other => panic!("未知 --blocks: {other}"),
        };
        let pred = logits.argmax(1).reshape([chunk.len()]);
        let pred_v: Vec<i64> = pred
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("读取预测失败");
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
        <B as burn::tensor::backend::Backend>::sync(device).expect("eval 逐批同步失败");
        <B as burn::tensor::backend::Backend>::memory_cleanup(device);
    }
    <B as burn::tensor::backend::Backend>::sync(device).expect("eval 同步失败");
    <B as burn::tensor::backend::Backend>::memory_cleanup(device);
    correct as f64 / n as f64 * 100.0
}

/// 零噪声因子（与槽布局匹配的全零张量）——无噪声锚点前向用
fn zero_factors(
    lora_slots: &[FSlotSpec],
    dense_slots: &[FSlotSpec],
    rank: usize,
    c: usize,
    device: &Dev,
) -> ChunkFactors {
    let lora = lora_slots
        .iter()
        .map(|s| {
            let [a, b] = s.shape.flat2d();
            (
                Tensor::<B, 3>::zeros([c, rank, a], device),
                Tensor::<B, 3>::zeros([c, rank, b], device),
            )
        })
        .collect();
    let dense = dense_slots
        .iter()
        .map(|s| {
            let n = s.shape.flat2d()[0] * s.shape.flat2d()[1];
            Tensor::<B, 2>::zeros([c, n], device)
        })
        .collect();
    ChunkFactors { lora, dense }
}

/// QamWeights → 零 delta QamRun（评估口径：m = (1+a)cos φ 基值、v_th 基值）
fn qam_run_zero(c: usize, qam_eval: Option<&QamWeights<B>>, device: &Dev) -> Option<QamRun<B>> {
    let q = qam_eval?;
    let mut sites = Vec::new();
    for (s, a, p) in &q.sites {
        let nn = a.dims()[0];
        let za = Tensor::<B, 2>::zeros([c, nn], device);
        let zp = Tensor::<B, 2>::zeros([c, nn], device);
        sites.push((
            *s,
            a.clone().reshape([nn, 1]),
            p.clone().reshape([nn, 1]),
            za,
            zp,
        ));
    }
    let vth = q
        .vth
        .iter()
        .map(|(s, v)| (*s, *v, Tensor::<B, 1>::zeros([c], device)))
        .collect();
    Some(QamRun { sites, vth })
}

/// 逐图像零噪声锚点 fitness b₀(x)（控制变量 X1）：
/// 当前 θ（含当前 QAM 基值 m）无噪声前向，逐图 loglik。
/// 与训练候选 fitness 同分布口径（同 relax/β/blocks），保证 E[raw′]=E[响应]。
#[allow(clippy::too_many_arguments)]
fn zero_noise_anchors(
    base: &SdtWeights<B>,
    lora_slots: &[FSlotSpec],
    dense_slots: &[FSlotSpec],
    qam_eval: Option<&QamWeights<B>>,
    data: &Cifar10Npz,
    cfg: &SdtConfig,
    blocks: &str,
    rank: usize,
    imgs: &[usize],
    batch: usize,
    t_steps: usize,
    relax_fwd: bool,
    relax_mlp: bool,
    beta: f32,
) -> Vec<f32> {
    let device = es_dev();
    let mut out = Vec::with_capacity(imgs.len());
    for chunk in imgs.chunks(batch) {
        let (images, targets) = data.get_batch(Split::Train, chunk, t_steps, device);
        let c = chunk.len();
        let factors = zero_factors(lora_slots, dense_slots, rank, c, device);
        let qam_run = qam_run_zero(c, qam_eval, device);
        let logits = match blocks {
            "conv" => es_forward_factored(images, base, &factors, cfg, relax_fwd, relax_mlp, beta, qam_run.as_ref()),
            "none" => es_forward_noconv(images, base, &factors, cfg, relax_fwd, relax_mlp, beta, qam_run.as_ref()),
            "probe" => es_forward_probe(images, base, &factors, cfg),
            other => panic!("未知 --blocks: {other}"),
        };
        let logsm = log_softmax2(logits);
        let raw = logsm
            .gather(1, targets.reshape([c, 1]))
            .reshape([c]);
        let v: Vec<f32> = raw
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("锚点读取失败");
        out.extend(v);
        <B as burn::tensor::backend::Backend>::sync(device).expect("锚点逐批同步失败");
        <B as burn::tensor::backend::Backend>::memory_cleanup(device);
    }
    out
}

/// ES 训练主入口（按 args.mode 分派）
pub fn run_train_es(args: EsArgs) {
    if args.mode == "factored" {
        return run_train_es_factored(args);
    }
    assert!(
        args.pop % 2 == 0,
        "pop 必须为偶数（反对称配对要求），实际 {}",
        args.pop
    );
    assert!(args.rank >= 1, "rank 必须 >= 1");
    let cfg = SdtConfig::default();
    let device = es_dev();

    // ---- 数据 ----
    let mut data_path = format!("{}/cifar10_data.npz", args.data_dir.trim_end_matches(['/', '\\']));
    if !std::path::Path::new(&data_path).exists() {
        data_path = "artifacts/cifar10_data.npz".to_string();
    }
    println!("== Burn SDT 演化策略（EggRoll-ES）训练 ==");
    println!(
        "数据: {data_path}（train={} test={}）T={} pop={} es_batch={} σ={} rank={} lr={} (AdamW)",
        "8000", "2000", args.time_steps, args.pop, args.es_batch, args.sigma, args.rank, args.lr
    );
    let data = Cifar10Npz::load(&data_path);

    // ---- 初始权重（与 SGD 基线同源：NPZ 融合权重 + 静态校准）----
    let start = std::time::Instant::now();
    let mut weights: SdtWeights<B> = match &args.weights {
        Some(w) if !w.is_empty() => {
            println!("加载初始化权重: {w}");
            let npz = crate::tensor_io::read_npz(w);
            crate::model::load_weights(&npz, &cfg, device)
        }
        _ => panic!("ES 训练需要初始权重 NPZ（--weights）"),
    };
    if !args.no_calibrate {
        let t0 = std::time::Instant::now();
        weights = crate::train::static_calibrate(&weights, &data, device, &cfg);
        println!(
            "[calibrate] 静态校准完成（耗时 {:.1}s）",
            t0.elapsed().as_secs_f32()
        );
    }
    // 校准后清池（与 train.rs 相同的顺序约束：先 sync 再 cleanup）
    <B as burn::tensor::backend::Backend>::sync(device).expect("GPU 同步失败");
    <B as burn::tensor::backend::Backend>::memory_cleanup(device);

    // ---- ES 装配 ----
    let slots = slots_of(&weights, cfg.depths);
    let mut params2d = flatten_params(&slots, &weights, cfg.depths);
    println!(
        "[es] 槽位={}（LoRA={} dense={}）参数量={}（不含 th_w）",
        slots.shapes.len(),
        slots.lora.iter().filter(|&x| *x).count(),
        slots.lora.iter().filter(|&x| !*x).count(),
        slots.num_params
    );

    let t0 = std::time::Instant::now();
    let deltas = build_delta_cache(&slots, &params2d, args.pop, args.sigma, args.rank, device);
    let cache_mb = args.pop * slots.num_params * 4 / (1024 * 1024);
    println!(
        "[es] 扰动缓存预生成完成：pop={} × 全部槽位（约 {cache_mb} MB VRAM，耗时 {:.2}s）",
        args.pop,
        t0.elapsed().as_secs_f32()
    );

    let mut optim = AdamW::new(&slots.shapes, args.lr as f32, 1e-4, device);

    // 每 epoch 的 generation 数（整除块，尾部丢弃）
    let gen_imgs = args.pop * args.es_batch;
    assert!(
        gen_imgs <= data.n_train,
        "pop×es_batch={} 超过训练集 {}（扰动缓存按 pop 物化，pop 过大还会爆显存）",
        gen_imgs,
        data.n_train
    );
    let gens_per_epoch = data.n_train / gen_imgs;
    let dropped = data.n_train % gen_imgs;
    println!(
        "[es] 每 epoch：{gens_per_epoch} 个 generation × {gen_imgs} 图（尾部丢弃 {dropped} 张），每代 1 次 AdamW 更新"
    );

    // CSV
    let mut csv_rows: Vec<String> =
        vec!["epoch,train_top1,train_loglik,val_top1,best_val,epoch_time,cum_time".to_string()];
    let mut best_val = 0.0_f64;
    let mut best_train = 0.0_f64;
    let mut cum_t = 0.0_f64;

    for epoch in 0..args.epochs {
        let ep_start = std::time::Instant::now();
        let order = crate::loader::shuffled_indices(data.n_train, args.seed + epoch as u64);
        let m = args.es_batch;

        let mut epoch_correct = 0usize;
        let mut epoch_imgs = 0usize;
        let mut epoch_loglik_sum = 0.0_f64;

        for gen in order.chunks_exact(gen_imgs) {
            // 1) 重建基础权重（原形状）
            let base_w = assemble(&slots, cfg.depths, &params2d);

            // 2) 逐候选：基础权重 + ΔW_c → 前向 → loglik/正确数 → 梯度累积
            let mut grad_acc: Vec<Tensor<B, 2>> = params2d
                .iter()
                .map(|p| Tensor::<B, 2>::zeros(p.dims(), device))
                .collect();
            let mut ones_acc: Vec<Tensor<B, 2>> = grad_acc.clone();
            let mut raws: Vec<Tensor<B, 1>> = Vec::with_capacity(args.pop);
            let mut corrects: Vec<Tensor<B, 1>> = Vec::with_capacity(args.pop);

            for c in 0..args.pop {
                let idx = &gen[c * m..(c + 1) * m];
                let (images, targets) =
                    data.get_batch(Split::Train, idx, args.time_steps, device);

                // 扰动权重组装（基于 base_w 的槽 + delta[c]）
                let pert = perturb_weights(&slots, cfg.depths, &base_w, &deltas[c]);
                let logits = forward_full(images, &pert, &cfg, None); // [m, C]

                // fitness：Σ log_softmax[label]（one-hot 加权和）
                let logsm = log_softmax2(logits.clone()); // [m, C]
                let onehot = one_hot(&data.train_y, idx, cfg.num_classes, device); // [m,C]
                let raw = (logsm * onehot).sum(); // [1]
                let pred = logits.argmax(1).reshape([m]);
                let correct = pred.equal(targets).float().sum(); // [1]

                // 梯度累积：Σ_c raw_c·ΔW_c 与 Σ_c ΔW_c（仿射 z-score 的两个充分统计量）
                for j in 0..grad_acc.len() {
                    let dw = deltas[c][j].clone();
                    grad_acc[j] = grad_acc[j].clone() + dw.clone() * raw.clone().reshape([1, 1]);
                    ones_acc[j] = ones_acc[j].clone() + dw;
                }
                raws.push(raw);
                corrects.push(correct);
            }

            // 3) 全局 z-score（仿射修正）→ 一次 AdamW 步
            let raws_t = Tensor::cat(raws, 0); // [pop]
            let mean = raws_t.clone().mean().into_scalar();
            let var = (raws_t.clone().powf_scalar(2.0).mean().into_scalar() - mean * mean).max(0.0);
            let stdv = (var + 1e-5).sqrt();
            let scale = -1.0 / (stdv * (args.pop as f32).sqrt());
            let grads: Vec<Tensor<B, 2>> = grad_acc
                .into_iter()
                .zip(ones_acc.into_iter())
                .map(|(g, o)| (g - o.mul_scalar(mean)).mul_scalar(scale))
                .collect();
            optim.step(&mut params2d, &grads);

            // 4) 统计（每代一次回读）
            let correct_t = Tensor::cat(corrects, 0).sum().into_scalar();
            epoch_correct += correct_t as usize;
            epoch_imgs += gen_imgs;
            // per-image loglik = mean_c(raw_c)/m（raw_c 为候选 c 的 m 张图 loglik 之和）
            epoch_loglik_sum += (mean / m as f32) as f64;
        }

        // 5) epoch 末：sync + cleanup（顺序约束与 train.rs 相同）
        <B as burn::tensor::backend::Backend>::sync(device).expect("GPU 同步失败");
        <B as burn::tensor::backend::Backend>::memory_cleanup(device);

        let train_top1 = epoch_correct as f64 / epoch_imgs.max(1) as f64 * 100.0;
        let train_loglik = epoch_loglik_sum / gens_per_epoch.max(1) as f64;
        if train_top1 > best_train {
            best_train = train_top1;
        }

        // 6) 验证
        let mut val_str = String::new();
        if epoch % args.validate_every == 0 || epoch == args.epochs - 1 {
            let w_now = assemble(&slots, cfg.depths, &params2d);
            let val_top1 = eval_es(&w_now, None, &data, 64, args.time_steps, &cfg);
            if val_top1 > best_val {
                best_val = val_top1;
            }
            val_str = format!("{val_top1:.2}");
            println!(
                "epoch={}/{}, train_top1={:.2}%, train_loglik={:.4}, val_top1={:.2}%, best_val={:.2}%（本轮耗时 {:.1}s）",
                epoch + 1,
                args.epochs,
                train_top1,
                train_loglik,
                val_top1,
                best_val,
                ep_start.elapsed().as_secs_f32()
            );
        } else {
            println!(
                "epoch={}/{}, train_top1={:.2}%, train_loglik={:.4}（本轮耗时 {:.1}s）",
                epoch + 1,
                args.epochs,
                train_top1,
                train_loglik,
                ep_start.elapsed().as_secs_f32()
            );
        }

        let el = ep_start.elapsed().as_secs_f64();
        cum_t += el;
        csv_rows.push(format!(
            "{},{:.2},{:.6},{},{:.2},{:.1},{:.1}",
            epoch + 1,
            train_top1,
            train_loglik,
            val_str,
            best_val,
            el,
            cum_t
        ));
        // 逐 epoch 落盘（长训练中断可保留部分结果）
        write_csv(&args.csv_out, &csv_rows);
    }

    println!(
        "ES 训练完成：best_val={best_val:.2}% best_train={best_train:.2}% 总耗时 {cum_t:.0}s"
    );
    println!("训练指标已写入: {}", args.csv_out);
    let _ = start;
}

/// 由标签构造 one-hot [m, C]（f32）
fn one_hot(labels: &[i64], idx: &[usize], num_classes: usize, device: &Dev) -> Tensor<B, 2> {
    let mut oh = vec![0.0_f32; idx.len() * num_classes];
    for (i, &si) in idx.iter().enumerate() {
        oh[i * num_classes + labels[si] as usize] = 1.0;
    }
    Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(oh, vec![idx.len(), num_classes]),
        device,
    )
}

/// 写 CSV（覆盖式）
fn write_csv(path: &str, rows: &[String]) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).expect("创建 CSV 目录失败");
        }
    }
    std::fs::write(path, rows.join("\n") + "\n").expect("写入 ES CSV 失败");
}

// ===========================================================================
// factored 模式：因式分解噪声前向（「小batch等效大batch」的完整形态）
// ===========================================================================
//
// cache 模式（上方）把每候选 ΔW 物化缓存，独立噪声方向数被 VRAM 卡在 ~800；
// 本模式复刻迁移版 `snn_transformer.rs::nn()` 的批量因式前向：
//
//     y = x·Wᵀ + σ'·((x·Bᵀ)·A)        （A [C,r,out] 已乘 ±σ_slot/√r，B [C,r,in]）
//
// 1×1 卷积本质是逐像素矩阵乘，可完全走该路径 → **每次更新的独立噪声方向数 =
// 全部训练图（默认 8000）**，与迁移版 batch=60000 的 hyperscale 形态同构。
// SPS 的 3×3 卷积无法因式分解 → 冻结（对应迁移版 freeze_nonlora 概念）；
// 可训练槽 = blocks 全部 6 个 1×1 卷积 ×2 + head（≈1.79M 参数，占总量 70%）。
// ΔW 不缓存，按 chunk 现场生成因子（CPU 并行 Box-Muller，反对称对只画一半），
// 每 chunk 前向后立即用同一批因子做梯度 einsum 累积（充分统计量），
// 代末一次全局 z-score 仿射修正 + 一次 AdamW（与 accumulate 架构严格一致）。

/// factored 槽位定位：blocks 6 个 1×1 卷积（0=q,1=k,2=v,3=proj,4=fc1,5=fc2）+ head
/// + QAM 调制槽（Qam(site, which)：which 0=a 1=φ，m=(1+a)·cos(φ)，ES_MANIFOLD §12）
/// + 可学习阈值槽（Vth(site)：每位点标量，混合估计器 §13——v_th 的对偶域坐标）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FLoc {
    BlkW(usize, usize),
    BlkB(usize, usize),
    HeadW,
    HeadB,
    Qam(usize, u8),
    Vth(usize),
    /// 普通 SNN（--blocks fc）：fc 层权重 [out,in]，k=0,1,2
    FcW(usize),
    /// 普通 SNN：fc 层偏置 [n]
    FcB(usize),
}

/// factored 槽规格（lora=权重走因式噪声；dense=偏置走加性噪声）
#[derive(Clone)]
struct FSlotSpec {
    loc: FLoc,
    key: u64,
    shape: OrigShape,
}

const F_WHICH: usize = 6; // 每块 6 个卷积

fn floc_get4(w: &SdtWeights<B>, blk: usize, which: usize) -> &Tensor<B, 4> {
    let b = &w.blk[blk];
    match which {
        0 => &b.q.w,
        1 => &b.k.w,
        2 => &b.v.w,
        3 => &b.proj.w,
        4 => &b.fc1.w,
        5 => &b.fc2.w,
        _ => unreachable!(),
    }
}

fn floc_get1(w: &SdtWeights<B>, blk: usize, which: usize) -> &Tensor<B, 1> {
    let b = &w.blk[blk];
    match which {
        0 => &b.q.b,
        1 => &b.k.b,
        2 => &b.v.b,
        3 => &b.proj.b,
        4 => &b.fc1.b,
        5 => &b.fc2.b,
        _ => unreachable!(),
    }
}

fn floc_set4(w: &mut SdtWeights<B>, blk: usize, which: usize, t: Tensor<B, 4>) {
    let b = &mut w.blk[blk];
    match which {
        0 => b.q.w = t,
        1 => b.k.w = t,
        2 => b.v.w = t,
        3 => b.proj.w = t,
        4 => b.fc1.w = t,
        5 => b.fc2.w = t,
        _ => unreachable!(),
    }
}

fn floc_set1(w: &mut SdtWeights<B>, blk: usize, which: usize, t: Tensor<B, 1>) {
    let b = &mut w.blk[blk];
    match which {
        0 => b.q.b = t,
        1 => b.k.b = t,
        2 => b.v.b = t,
        3 => b.proj.b = t,
        4 => b.fc1.b = t,
        5 => b.fc2.b = t,
        _ => unreachable!(),
    }
}

/// 构造 factored 槽位（顺序固定：lora 0..12 = blk{0,1}×{q,k,v,proj,fc1,fc2}·w + head_w；
/// dense 0..12 = 对应偏置 + head_b）
fn factored_slots(
    base: &SdtWeights<B>,
    depths: usize,
    qam_sites: &[usize],
    vth_sites: &[usize],
    blocks: &str,
) -> (Vec<FSlotSpec>, Vec<FSlotSpec>) {
    let mut lora = Vec::new();
    let mut dense = Vec::new();
    let mut ki = 0u64;
    if blocks == "fc" {
        // 普通 SNN（HyperscaleES 主场）：fc1[128,3072]→LIF→fc2[128,128]→LIF→fc3[10,128]
        // + 偏置×3 + QAM（LIF1/LIF2 输入电流调制，各 128 神经元）。基底随机初始化。
        const FC_HIDE: usize = 128;
        lora.push(FSlotSpec { loc: FLoc::FcW(0), key: (ki + 1).wrapping_mul(KEY_MUL), shape: OrigShape::W2([FC_HIDE, 3072]) });
        ki += 1;
        lora.push(FSlotSpec { loc: FLoc::FcW(1), key: (ki + 1).wrapping_mul(KEY_MUL), shape: OrigShape::W2([FC_HIDE, FC_HIDE]) });
        ki += 1;
        lora.push(FSlotSpec { loc: FLoc::FcW(2), key: (ki + 1).wrapping_mul(KEY_MUL), shape: OrigShape::W2([10, FC_HIDE]) });
        ki += 1;
        dense.push(FSlotSpec { loc: FLoc::FcB(0), key: (ki + 1).wrapping_mul(KEY_MUL), shape: OrigShape::B1(FC_HIDE) });
        ki += 1;
        dense.push(FSlotSpec { loc: FLoc::FcB(1), key: (ki + 1).wrapping_mul(KEY_MUL), shape: OrigShape::B1(FC_HIDE) });
        ki += 1;
        dense.push(FSlotSpec { loc: FLoc::FcB(2), key: (ki + 1).wrapping_mul(KEY_MUL), shape: OrigShape::B1(10) });
        ki += 1;
        for &site in qam_sites {
            dense.push(FSlotSpec { loc: FLoc::Qam(site, 0), key: (ki + 1).wrapping_mul(KEY_MUL), shape: OrigShape::B1(FC_HIDE) });
            ki += 1;
            dense.push(FSlotSpec { loc: FLoc::Qam(site, 1), key: (ki + 1).wrapping_mul(KEY_MUL), shape: OrigShape::B1(FC_HIDE) });
            ki += 1;
        }
        return (lora, dense);
    }
    if blocks != "conv" {
        // none/probe：无块卷积槽；none 保留 head+QAM（+vth），probe 只留 head
        lora.push(FSlotSpec {
            loc: FLoc::HeadW,
            key: (ki + 1).wrapping_mul(KEY_MUL),
            shape: OrigShape::W2(base.head_w.dims()),
        });
        ki += 1;
        dense.push(FSlotSpec {
            loc: FLoc::HeadB,
            key: (ki + 1).wrapping_mul(KEY_MUL),
            shape: OrigShape::B1(base.head_b.dims()[0]),
        });
        ki += 1;
        if blocks == "none" {
            const QAM_SITE_N: [usize; 8] = [256, 256, 256, 256, 256, 256, 1024, 256];
            for &site in qam_sites {
                dense.push(FSlotSpec {
                    loc: FLoc::Qam(site, 0),
                    key: (ki + 1).wrapping_mul(KEY_MUL),
                    shape: OrigShape::B1(QAM_SITE_N[site]),
                });
                ki += 1;
                dense.push(FSlotSpec {
                    loc: FLoc::Qam(site, 1),
                    key: (ki + 1).wrapping_mul(KEY_MUL),
                    shape: OrigShape::B1(QAM_SITE_N[site]),
                });
                ki += 1;
            }
        }
        return (lora, dense);
    }
    for j in 0..depths {
        for which in 0..F_WHICH {
            let dims = floc_get4(base, j, which).dims();
            lora.push(FSlotSpec {
                loc: FLoc::BlkW(j, which),
                key: (ki + 1).wrapping_mul(KEY_MUL),
                shape: OrigShape::W4(dims),
            });
            ki += 1;
            let bd = floc_get1(base, j, which).dims()[0];
            dense.push(FSlotSpec {
                loc: FLoc::BlkB(j, which),
                key: (ki + 1).wrapping_mul(KEY_MUL),
                shape: OrigShape::B1(bd),
            });
            ki += 1;
        }
    }
    lora.push(FSlotSpec {
        loc: FLoc::HeadW,
        key: (ki + 1).wrapping_mul(KEY_MUL),
        shape: OrigShape::W2(base.head_w.dims()),
    });
    ki += 1;
    dense.push(FSlotSpec {
        loc: FLoc::HeadB,
        key: (ki + 1).wrapping_mul(KEY_MUL),
        shape: OrigShape::B1(base.head_b.dims()[0]),
    });
    ki += 1;
    // QAM 调制槽（位点神经元数：xs/q/k/v/kv/x1/feat=256，x2=1024）
    const QAM_SITE_N: [usize; 8] = [256, 256, 256, 256, 256, 256, 1024, 256];
    for &site in qam_sites {
        dense.push(FSlotSpec {
            loc: FLoc::Qam(site, 0),
            key: (ki + 1).wrapping_mul(KEY_MUL),
            shape: OrigShape::B1(QAM_SITE_N[site]),
        });
        ki += 1;
        dense.push(FSlotSpec {
            loc: FLoc::Qam(site, 1),
            key: (ki + 1).wrapping_mul(KEY_MUL),
            shape: OrigShape::B1(QAM_SITE_N[site]),
        });
        ki += 1;
    }
    // 可学习阈值槽（混合估计器 §13）：每位点标量 B1(1)
    for &site in vth_sites {
        dense.push(FSlotSpec {
            loc: FLoc::Vth(site),
            key: (ki + 1).wrapping_mul(KEY_MUL),
            shape: OrigShape::B1(1),
        });
        ki += 1;
    }
    (lora, dense)
}

/// 一个 chunk 的噪声因子（已上传 GPU）
struct ChunkFactors {
    /// per lora 槽：(A [C,r,out] 已乘 ±σ_slot/√r, B [C,r,in])
    lora: Vec<(Tensor<B, 3>, Tensor<B, 3>)>,
    /// per dense 槽：[C, numel]（±σ_slot·N(0,1)）
    dense: Vec<Tensor<B, 2>>,
}

/// CPU 端生成一个 chunk 的全部噪声 flat buffer（只含 Send 数据，可在后台线程运行；
/// 与 GPU 前向流水线重叠）。返回 (lora flat per slot, dense flat per slot)。
/// lora 槽行布局：每候选 (a+b)×r 标准正态（前 b 行=B，后 a 行=A×sign·σ_slot/√r），
/// 与迁移版 get_lora_update_params 的切分约定一致；反对称对共享随机流。
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // S3 后 factored 路径改用 gen_chunk_factors_gpu；保留供 cache 模式对照
fn gen_chunk_flat(
    lora_slots: Vec<FSlotSpec>,
    dense_slots: Vec<FSlotSpec>,
    sigma_lora: Vec<f32>,
    sigma_dense: Vec<f32>,
    rank: usize,
    cand: Vec<usize>,
    epoch: i32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let c = cand.len();
    // ---- CPU 并行生成：每线程一段候选，各自产出 per-slot flat buffer ----
    let cores = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    let n_threads = cores.min(c.max(1));
    let per = c.div_ceil(n_threads);
    let (lora_parts, dense_parts): (Vec<Vec<f32>>, Vec<Vec<f32>>) =
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..n_threads {
                let lo = t * per;
                let hi = ((t + 1) * per).min(c);
                let lora_slots_ref = &lora_slots;
                let dense_slots_ref = &dense_slots;
                let sigma_lora_ref = &sigma_lora;
                let sigma_dense_ref = &sigma_dense;
                let cand_ref = &cand;
                handles.push(s.spawn(move || {
                    let mut lb: Vec<Vec<f32>> = lora_slots_ref
                        .iter()
                        .map(|slot| {
                            let [a, b] = slot.shape.flat2d();
                            vec![0.0_f32; (hi - lo) * (a + b) * rank]
                        })
                        .collect();
                    let mut db: Vec<Vec<f32>> = dense_slots_ref
                        .iter()
                        .map(|slot| {
                            let n = slot.shape.flat2d()[0] * slot.shape.flat2d()[1];
                            vec![0.0_f32; (hi - lo) * n]
                        })
                        .collect();
                    for (local, &g) in cand_ref.iter().enumerate().take(hi).skip(lo) {
                        let row = local - lo;
                        let sign = if g % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
                        let pair = (g / 2) as i32;
                        for (si, slot) in lora_slots_ref.iter().enumerate() {
                            let [a, b] = slot.shape.flat2d();
                            let base_sigma = sigma_lora_ref[si] / (rank as f32).sqrt();
                            let mut rng =
                                DeterministicNoise::new(noise_seed(slot.key, epoch, pair));
                            let row_off = row * (a + b) * rank;
                            // B 区（前 b 行）+ A 区（后 a 行，乘 sign·base_sigma）
                            for k in 0..(a + b) * rank {
                                lb[si][row_off + k] = rng.standard_normal();
                            }
                            for k in 0..a * rank {
                                lb[si][row_off + b * rank + k] *= sign * base_sigma;
                            }
                        }
                        for (si, slot) in dense_slots_ref.iter().enumerate() {
                            let n = slot.shape.flat2d()[0] * slot.shape.flat2d()[1];
                            let mut rng =
                                DeterministicNoise::new(noise_seed(slot.key, epoch, pair));
                            let off = row * n;
                            for k in 0..n {
                                db[si][off + k] = rng.standard_normal() * sign * sigma_dense_ref[si];
                            }
                        }
                    }
                    (lb, db)
                }));
            }
            let mut lora_all: Vec<Vec<Vec<f32>>> = (0..lora_slots.len()).map(|_| Vec::new()).collect();
            let mut dense_all: Vec<Vec<Vec<f32>>> = (0..dense_slots.len()).map(|_| Vec::new()).collect();
            for h in handles {
                let (lb, db) = h.join().unwrap();
                for (si, part) in lb.into_iter().enumerate() {
                    lora_all[si].push(part);
                }
                for (si, part) in db.into_iter().enumerate() {
                    dense_all[si].push(part);
                }
            }
            // 按 chunk 内候选顺序拼接各线程分段
            let flat_lora: Vec<Vec<f32>> = lora_all
                .into_iter()
                .map(|mut parts| {
                    let mut all = Vec::new();
                    for p in parts.drain(..) {
                        all.extend(p);
                    }
                    all
                })
                .collect();
            let flat_dense: Vec<Vec<f32>> = dense_all
                .into_iter()
                .map(|mut parts| {
                    let mut all = Vec::new();
                    for p in parts.drain(..) {
                        all.extend(p);
                    }
                    all
                })
                .collect();
            (flat_lora, flat_dense)
        });

    (lora_parts, dense_parts)
}

/// 上传一个 chunk 的 flat 噪声并切分 A/B（主线程执行，~155MB PCIe）
#[allow(dead_code)] // S3 后 factored 路径不再上传；保留供 cache 模式对照
fn upload_chunk_factors(
    lora_parts: Vec<Vec<f32>>,
    dense_parts: Vec<Vec<f32>>,
    lora_slots: &[FSlotSpec],
    dense_slots: &[FSlotSpec],
    rank: usize,
    c: usize,
    device: &Dev,
) -> ChunkFactors {
    // ---- 上传 + 切分 A/B ----
    let mut lora_out = Vec::with_capacity(lora_slots.len());
    for (si, slot) in lora_slots.iter().enumerate() {
        let [a, b] = slot.shape.flat2d();
        let lora_t = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(
                lora_parts[si].clone(),
                vec![c, a + b, rank],
            ),
            device,
        );
        let a_t = lora_t
            .clone()
            .slice([0..c, b..(a + b), 0..rank])
            .swap_dims(1, 2); // [C, r, out]
        let b_t = lora_t.slice([0..c, 0..b, 0..rank]).swap_dims(1, 2); // [C, r, in]
        lora_out.push((a_t, b_t));
    }
    let mut dense_out = Vec::with_capacity(dense_slots.len());
    for (si, slot) in dense_slots.iter().enumerate() {
        let n = slot.shape.flat2d()[0] * slot.shape.flat2d()[1];
        dense_out.push(Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(dense_parts[si].clone(), vec![c, n]),
            device,
        ));
    }
    ChunkFactors {
        lora: lora_out,
        dense: dense_out,
    }
}

/// S3：GPU 端确定性噪声因子生成（xorshift32 + Box-Muller 全向量化）。
/// 因子直接在显存构建——零 CPU 生成、零 PCIe 上传（原 ~9.6GB/epoch）。
/// 语义与 CPU 版一致：候选 g 的 seed = noise_seed(key, epoch, g/2)，
/// B 区（前 b 行）共享原始法向，A 区（后 a 行）×(sign(g)·σ_slot/√r)，dense ±σ_slot。
/// 注意：RNG 流值与 CPU 版不同（xorshift32-GPU vs xorshift64-CPU）⇒ 与旧结果对比需重基线。
fn gen_chunk_factors_gpu(
    lora_slots: &[FSlotSpec],
    dense_slots: &[FSlotSpec],
    sigma_lora: &[f32],
    sigma_dense: &[f32],
    rank: usize,
    cand_slice: &[usize],
    epoch: i32,
    device: &Dev,
) -> ChunkFactors {
    let c = cand_slice.len();
    let sign_t = Tensor::<B, 1>::from_floats(
        cand_slice
            .iter()
            .map(|&g| if g % 2 == 0 { 1.0f32 } else { -1.0f32 })
            .collect::<Vec<f32>>()
            .as_slice(),
        device,
    )
    .reshape([c, 1]);
    // 按唯一 count 分组生成（同形状槽共用一次 kernel 链，削减启动数）
    let mut lora_out = Vec::with_capacity(lora_slots.len());
    lora_out.resize_with(lora_slots.len(), || {
        (
            Tensor::<B, 3>::zeros([1, 1, 1], device),
            Tensor::<B, 3>::zeros([1, 1, 1], device),
        )
    });
    for (group, sis) in lora_groupby_count(lora_slots, rank) {
        let l = group;
        // 种子 [n, c, 1]：n 个槽 × c 候选
        let mut seeds = Vec::with_capacity(sis.len() * c);
        for &si in &sis {
            for &g in cand_slice {
                let s = noise_seed(lora_slots[si].key, epoch, (g / 2) as i32);
                seeds.push(((s ^ (s >> 32)) as u32) as i32);
            }
        }
        let n = sis.len();
        let seed_t = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(seeds, [n * c]),
            device,
        )
        .reshape([n, c, 1]);
        let normals = gpu_normals3(seed_t, l, device); // [n, c, l]
        for (gi, &si) in sis.iter().enumerate() {
            let [a, b] = lora_slots[si].shape.flat2d();
            let nm = normals.clone().slice([gi..gi + 1]).reshape([c, l]); // [c, l]
            // B 区共享原始法向（pair 内不反号）
            let b_part = nm.clone().narrow(1, 0, b * rank).reshape([c, b, rank]);
            // A 区 ×(sign·σ/√r)
            let a_part = nm.narrow(1, b * rank, a * rank).reshape([c, a, rank])
                * sign_t
                    .clone()
                    .reshape([c, 1, 1])
                    .mul_scalar(sigma_lora[si] / (rank as f32).sqrt());
            let lora_t = Tensor::cat(vec![b_part, a_part], 1); // [c, a+b, rank]
            let a_t = lora_t
                .clone()
                .slice([0..c, b..(a + b), 0..rank])
                .swap_dims(1, 2);
            let b_t = lora_t.slice([0..c, 0..b, 0..rank]).swap_dims(1, 2);
            lora_out[si] = (a_t, b_t);
        }
    }
    let mut dense_out = Vec::with_capacity(dense_slots.len());
    dense_out.resize_with(dense_slots.len(), || Tensor::<B, 2>::zeros([1, 1], device));
    for (group, sis) in dense_groupby_count(dense_slots) {
        let n_dim = group;
        let mut seeds = Vec::with_capacity(sis.len() * c);
        for &si in &sis {
            for &g in cand_slice {
                let s = noise_seed(dense_slots[si].key, epoch, (g / 2) as i32);
                seeds.push(((s ^ (s >> 32)) as u32) as i32);
            }
        }
        let n = sis.len();
        let seed_t = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(seeds, [n * c]),
            device,
        )
        .reshape([n, c, 1]);
        let normals = gpu_normals3(seed_t, n_dim, device); // [n, c, count]
        for (gi, &si) in sis.iter().enumerate() {
            let nm = normals.clone().slice([gi..gi + 1]).reshape([c, n_dim]);
            let d2 = nm * sign_t.clone().mul_scalar(sigma_dense[si]);
            dense_out[si] = d2;
        }
    }
    ChunkFactors {
        lora: lora_out,
        dense: dense_out,
    }
}

/// lora 槽按唯一 l=(a+b)·rank 分组（保持槽序）
fn lora_groupby_count(lora_slots: &[FSlotSpec], rank: usize) -> Vec<(usize, Vec<usize>)> {
    let mut order: Vec<usize> = Vec::new();
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (si, slot) in lora_slots.iter().enumerate() {
        let [a, b] = slot.shape.flat2d();
        let l = (a + b) * rank;
        if let Some(g) = groups.iter_mut().find(|(lg, _)| *lg == l) {
            g.1.push(si);
        } else {
            order.push(l);
            groups.push((l, vec![si]));
        }
    }
    let _ = order;
    groups
}

/// dense 槽按唯一元素数分组（保持槽序）
fn dense_groupby_count(dense_slots: &[FSlotSpec]) -> Vec<(usize, Vec<usize>)> {
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (si, slot) in dense_slots.iter().enumerate() {
        let n = slot.shape.flat2d()[0] * slot.shape.flat2d()[1];
        if let Some(g) = groups.iter_mut().find(|(ng, _)| *ng == n) {
            g.1.push(si);
        } else {
            groups.push((n, vec![si]));
        }
    }
    groups
}

/// [n, c, count] 标准法向（rank-3 批量版）：
/// state = seed ⊕ fmix32(elem·GOLDEN)（murmur3 充分雪崩，消除 elem 与 elem+half
/// 的常数 XOR 关联——xorshift 是 GF(2) 线性映射，弱混合会让 Box-Muller 两半
/// 确定性相关 → 法向分布系统偏差），fmix32 + xorshift32×2，Box-Muller 两半各一组。
fn gpu_normals3(seed_t: Tensor<B, 3, Int>, count: usize, device: &Dev) -> Tensor<B, 3> {
    let half_count = count.div_ceil(2);
    let elem = Tensor::<B, 1, Int>::arange(0..(half_count * 2) as i64, device)
        .reshape([1, 1, half_count * 2]);
    // fmix32（murmur3）：h ^= h>>16; h *= 0x21f0aaad; h ^= h>>15; h *= 0x735a2d97; h ^= h>>15
    let z0 = seed_t
        .clone()
        .bitwise_xor(elem.bitwise_xor_scalar((-1640531527i32))); // elem·GOLDEN（u32 黄金率回绕为 i32）
    let mut z = z0;
    z = z.clone().bitwise_xor(z.clone().bitwise_right_shift_scalar(16i32));
    z = z * 0x21F0_AAADi32;
    z = z.clone().bitwise_xor(z.clone().bitwise_right_shift_scalar(15i32));
    z = z * 0x735A_2D97i32;
    z = z.clone().bitwise_xor(z.clone().bitwise_right_shift_scalar(15i32));
    // xorshift32 ×2
    z = z.clone().bitwise_xor(z.clone().bitwise_left_shift_scalar(13i32));
    z = z.clone().bitwise_xor(z.clone().bitwise_right_shift_scalar(17i32));
    z = z.clone().bitwise_xor(z.clone().bitwise_left_shift_scalar(5i32));
    z = z.clone().bitwise_xor(z.clone().bitwise_left_shift_scalar(13i32));
    z = z.clone().bitwise_xor(z.clone().bitwise_right_shift_scalar(17i32));
    z = z.clone().bitwise_xor(z.clone().bitwise_left_shift_scalar(5i32));
    let u = z
        .bitwise_right_shift_scalar(8i32)
        .bitwise_and_scalar(0x00FF_FFFFi32)
        .float()
        .div_scalar(16777216.0); // [0,1)
    let u1 = u.clone().narrow(2, 0, half_count).clamp(1e-6, 1.0);
    let u2 = u.narrow(2, half_count, half_count);
    let r = (u1.log().mul_scalar(-2.0)).sqrt();
    let two_pi_u2 = u2.mul_scalar(2.0 * std::f32::consts::PI);
    let n1 = r.clone() * two_pi_u2.clone().cos();
    let n2 = r * two_pi_u2.sin();
    Tensor::cat(vec![n1, n2], 2).narrow(2, 0, count)
}

/// 因式分解噪声的 1×1 卷积（批量候选并行，完全复刻迁移版 nn() 语义）。
///
/// `x` [T, Cb, Kin, H, W]（Cb=C·es_batch，factored 模式 es_batch=1 → Cb=C）；
/// `w2d` [Kout, Kin] 基底；`bias5` [1, Cb, Kout, 1, 1]（基底偏置+逐候选偏置噪声）；
/// 返回 [T, Cb, Kout, H, W]。
fn noisy_conv1x1(
    x: Tensor<B, 5>,
    w2d: &Tensor<B, 2>,
    bias5: Tensor<B, 5>,
    a_t: &Tensor<B, 3>,
    b_t: &Tensor<B, 3>,
) -> Tensor<B, 5> {
    let d = x.dims();
    let (t, cb, kin, h, w) = (d[0], d[1], d[2], d[3], d[4]);
    // [Cb, T·H·W, Kin]（候选维提前，逐像素行展开）
    let x3 = x
        .swap_dims(0, 1) // [Cb, T, Kin, H, W]
        .permute([0, 1, 3, 4, 2]) // [Cb, T, H, W, Kin]
        .reshape([cb, t * h * w, kin]);
    // 基底：共享 W 的 2D GEMM（1x1 conv 等价形式）
    let base = x3
        .clone()
        .reshape([cb * t * h * w, kin])
        .matmul(w2d.clone().transpose())
        .reshape([cb, t * h * w, w2d.dims()[0]]);
    // 噪声：批量 (x@Bᵀ)@A —— (Cb,M,Kin)@(Cb,Kin,r) -> (Cb,M,r) @ (Cb,r,Kout)
    let yb = x3.matmul(b_t.clone().swap_dims(1, 2));
    let yn = yb.matmul(a_t.clone());
    let out3 = base + yn; // [Cb, THW, Kout]
    out3.reshape([cb, t, h, w, w2d.dims()[0]])
        .permute([1, 0, 4, 2, 3]) // [T, Cb, Kout, H, W]
        + bias5
}

/// LoRA 槽梯度 einsum：Σ_c s_c·(A_c·B_cᵀ)（weighted=false 时 s=1，即 ones 项）。
/// A [C,r,out]、B [C,r,in] 连续 → 展平 [C·r, ·] 后 2D GEMM（迁移版 2D gemm 等价式）。
fn lora_grad_einsum(
    a: &Tensor<B, 3>,
    b: &Tensor<B, 3>,
    s: &Tensor<B, 1>,
    weighted: bool,
) -> Tensor<B, 2> {
    let [c, r, out] = a.dims();
    let din = b.dims()[2];
    let a_w = if weighted {
        a.clone() * s.clone().reshape([c, 1, 1])
    } else {
        a.clone()
    };
    let a_flat = a_w.reshape([c * r, out]);
    let b_flat = b.clone().reshape([c * r, din]);
    a_flat.transpose().matmul(b_flat) // [out, in]
}

/// QAM 运行时（逐候选扰动版，ES_MANIFOLD §12）：每启用位点
/// (site_id, base_a [n,1], base_phi [n,1], δa [C,n], δφ [C,n])，
/// m_c = (1 + (a + δa_c)) · cos(φ + δφ_c) ∈ [C, n]。
/// vth：每启用位点 (site_id, 基值 f64, δ [C])——混合估计器 §13 的逐候选阈值扰动。
struct QamRun<B: Backend> {
    sites: Vec<(usize, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>,
    vth: Vec<(usize, f64, Tensor<B, 1>)>,
}

impl<B: Backend> QamRun<B> {
    /// 逐候选调制 [C, n]（存在该位点时）
    fn m(&self, site: usize) -> Option<Tensor<B, 2>> {
        for (sid, a, phi, da, dphi) in &self.sites {
            if *sid == site {
                let n = a.dims()[0];
                let a_c = a.clone().reshape([1, n]) + da.clone(); // [C, n]
                let phi_c = phi.clone().reshape([1, n]) + dphi.clone();
                let m = (a_c + 1.0) * phi_c.cos();
                return Some(m);
            }
        }
        None
    }

    /// 逐候选阈值张量（rank5 形 [1,C,1,1,1]；不存在该位点时返回 None）
    fn th5(&self, site: usize) -> Option<Tensor<B, 5>> {
        for (sid, base, dt) in &self.vth {
            if *sid == site {
                let c = dt.dims()[0];
                return Some((dt.clone() + *base as f32).reshape([1, c, 1, 1, 1]));
            }
        }
        None
    }

    /// 逐候选阈值张量（rank4 形 [1,C,1,1]，feat 位点用）
    fn th4(&self, site: usize) -> Option<Tensor<B, 4>> {
        for (sid, base, dt) in &self.vth {
            if *sid == site {
                let c = dt.dims()[0];
                return Some((dt.clone() + *base as f32).reshape([1, c, 1, 1]));
            }
        }
        None
    }
}

/// factored 前向：SPS 冻结（共享权重）+ blocks/head 因式噪声。
/// relax=true 时所有 LIF 读出换 σ(β(h−thr))（TSES 温度松弛，定理 3）。
/// 返回 [Cb, num_classes] logits（时间平均后）。
fn es_forward_factored(
    images: Tensor<B, 5>, // [T, C, 3, 32, 32]
    base: &SdtWeights<B>,
    f: &ChunkFactors,
    cfg: &SdtConfig,
    relax_ssa: bool,
    relax_mlp: bool,
    beta: f32,
    qam: Option<&QamRun<B>>,
) -> Tensor<B, 2> {
    let t = images.dims()[0];
    let c = images.dims()[1];
    let device = es_dev();
    // S6：松弛作用域——ssa 模式只松弛 SSA 位点（xs/q/k/v/kv，梯度信号集中处），
    // MLP 位点（x1/x2）与 head 用硬 LIF；all 模式全部松弛（与 v4 行为一致）
    // 位点感知 LIF（§13 混合估计器）：该位点有 vth 扰动槽时用逐候选阈值张量
    // （±σ·ε 加到基值上、形 [1,C,1,1,1] 广播），否则退回标量阈值路径
    let lif = |x: Tensor<B, 5>, site: usize, th: f64| -> Tensor<B, 5> {
        match qam.and_then(|q| q.th5(site)) {
            Some(th_t) => lif_seq_bth(x, th_t, if relax_ssa { Some(beta as f64) } else { None }),
            None => {
                if relax_ssa {
                    lif_seq_relaxed(x, th, beta as f64)
                } else {
                    lif_seq(x, th)
                }
            }
        }
    };
    let lif_mlp = |x: Tensor<B, 5>, site: usize, th: f64| -> Tensor<B, 5> {
        match qam.and_then(|q| q.th5(site)) {
            Some(th_t) => lif_seq_bth(x, th_t, if relax_mlp { Some(beta as f64) } else { None }),
            None => {
                if relax_mlp {
                    lif_seq_relaxed(x, th, beta as f64)
                } else {
                    lif_seq(x, th)
                }
            }
        }
    };
    let lif4 = |x: Tensor<B, 4>, site: usize, th: f64| -> Tensor<B, 4> {
        match qam.and_then(|q| q.th4(site)) {
            Some(th_t) => lif_seq_bth(x, th_t, if relax_mlp { Some(beta as f64) } else { None }),
            None => {
                if relax_mlp {
                    lif_seq_relaxed(x, th, beta as f64)
                } else {
                    lif_seq(x, th)
                }
            }
        }
    };

    // SPS（冻结，共享权重，候选并入 batch 维）
    let mut x = forward_sps(images, base, cfg);

    // blocks：SSA + MLP（q/k/v/proj/fc1/fc2 全部走因式噪声 1×1 conv）
    for j in 0..base.blk.len() {
        let d = x.dims();
        let (_t, _b, ch, hh, ww) = (d[0], d[1], d[2], d[3], d[4]);
        let heads = cfg.num_heads;
        let head_dim = ch / heads;

        // 槽位索引：lora[j*6+which]，dense 同序
        let lw = |which: usize| &f.lora[j * F_WHICH + which];
        let db = |which: usize| &f.dense[j * F_WHICH + which];

        // SSA
        // xs 位点 0：m [C,256] 调制 SPS 输出
        let xs = {
            let xin = match qam.and_then(|q| q.m(0)) {
                Some(m) => x.clone() * m.reshape([1, c, 256, 1, 1]),
                None => x.clone(),
            };
            lif(xin, 0, 1.0)
        };
        let identity = x;
        // 偏置合并项：[1, C, Kout, 1, 1] = b_base + 逐候选噪声
        let bias5 = |which: usize, kout: usize| -> Tensor<B, 5> {
            let b_base = flatten_slot(&OrigShape::B1(kout), floc_get1(base, j, which));
            (b_base.reshape([1, kout]) + db(which).clone()).reshape([1, c, kout, 1, 1])
        };
        let w2 = |which: usize| -> Tensor<B, 2> {
            let dims = floc_get4(base, j, which).dims();
            flatten_slot(
                &OrigShape::W4(dims),
                floc_get4(base, j, which),
            )
        };
        let xq = noisy_conv1x1(xs.clone(), &w2(0), bias5(0, ch), &lw(0).0, &lw(0).1);
        let xk = noisy_conv1x1(xs.clone(), &w2(1), bias5(1, ch), &lw(1).0, &lw(1).1);
        let xv = noisy_conv1x1(xs, &w2(2), bias5(2, ch), &lw(2).0, &lw(2).1);

        // q/k/v 位点 1/2/3：m [C,256] 调制 conv 输出
        let q = {
            let xin = match qam.and_then(|q| q.m(1)) {
                Some(m) => xq * m.reshape([1, c, 256, 1, 1]),
                None => xq,
            };
            lif(xin, 1, 1.0)
        };
        let k = {
            let xin = match qam.and_then(|q| q.m(2)) {
                Some(m) => xk * m.reshape([1, c, 256, 1, 1]),
                None => xk,
            };
            lif(xin, 2, 1.0)
        };
        let v = {
            let xin = match qam.and_then(|q| q.m(3)) {
                Some(m) => xv * m.reshape([1, c, 256, 1, 1]),
                None => xv,
            };
            lif(xin, 3, 1.0)
        };

        let qh = reshape_heads(q, heads, head_dim);
        let kh = reshape_heads(k, heads, head_dim);
        let vh = reshape_heads(v, heads, head_dim);

        let kv = (kh * vh.clone()).sum_dim(3);
        // kv 位点 4：m [C,256] reshape [1,C,heads,1,hd]
        let kv_spike = {
            let kin = match qam.and_then(|q| q.m(4)) {
                Some(m) => kv * m.reshape([1, c, heads, 1, head_dim]),
                None => kv,
            };
            lif(kin, 4, 0.5)
        };
        let xattn = qh * kv_spike;
        let xo = unshape_heads(xattn, heads, head_dim, hh, ww);

        let xp = noisy_conv1x1(xo, &w2(3), bias5(3, ch), &lw(3).0, &lw(3).1);
        let ssa_out = xp + identity;

        // MLP
        let identity2 = ssa_out.clone();
        // x1 位点 5：m [C,256] 调制 ssa_out
        let x1 = {
            let xin = match qam.and_then(|q| q.m(5)) {
                Some(m) => ssa_out * m.reshape([1, c, 256, 1, 1]),
                None => ssa_out,
            };
            lif_mlp(xin, 5, 1.0)
        };
        let hidden = cfg.mlp_hidden();
        let x1c = noisy_conv1x1(x1, &w2(4), bias5(4, hidden), &lw(4).0, &lw(4).1);
        // x2 位点 6：m [C,1024] 调制 fc1 conv 输出
        let x2 = {
            let xin = match qam.and_then(|q| q.m(6)) {
                Some(m) => x1c * m.reshape([1, c, hidden, 1, 1]),
                None => x1c,
            };
            lif_mlp(xin, 6, 1.0)
        };
        let x2c = noisy_conv1x1(x2, &w2(5), bias5(5, ch), &lw(5).0, &lw(5).1);
        x = x2c + identity2;
    }

    // head：flatten(3).mean(3) → LIF → 因式噪声 Linear → 时间平均
    let fd = x.dims();
    let (t, cb, ch) = (fd[0], fd[1], fd[2]);
    let feat = x
        .reshape([t, cb, ch, fd[3] * fd[4]])
        .sum_dim(3)
        .div_scalar((fd[3] * fd[4]) as f32);
    // feat 位点 7：m [C,256]（紧邻 head 读出——位点深度律最高信号槽）
    // feat 实为 rank-4 [T,C,ch,1]（sum_dim 保留维）
    let feat = match qam.and_then(|q| q.m(7)) {
        Some(m) => feat * m.reshape([1, c, ch, 1]),
        None => feat,
    };
    let feat_spike = lif4(feat, 7, 1.0); // [T, C, 256, 1]（sum_dim 保留维）
    let f3 = feat_spike.permute([1, 0, 2, 3]).reshape([c, t, ch]); // [C, T, 256]
    let head_idx = base.blk.len() * F_WHICH; // lora/dense 的 head 槽索引
    let (a_h, b_h) = &f.lora[head_idx];
    let head_w2 = base.head_w.clone(); // [10, 256]
    let base_h = f3
        .clone()
        .reshape([cb * t, ch])
        .matmul(head_w2.transpose())
        .reshape([cb, t, cfg.num_classes]);
    let yn_h = f3.matmul(b_h.clone().swap_dims(1, 2)).matmul(a_h.clone()); // [C, T, 10]
    let hb = flatten_slot(&OrigShape::B1(cfg.num_classes), &base.head_b); // [10,1]
    let bias3 = (hb.reshape([1, cfg.num_classes]) + f.dense[head_idx].clone()) // [C, 10]
        .reshape([c, 1, cfg.num_classes]); // [C, 1, 10]（对 T 广播）
    let logits3 = base_h + yn_h + bias3; // [C, T, 10]
    let _ = device;
    logits3
        .sum_dim(1)
        .squeeze::<2>()
        .div_scalar(t as f32) // [C, 10]（时间平均）
}

/// 无卷积前向（--blocks none）：块内六个 1×1 conv 全部恒等直连，
/// 注意力核直接作用于 SPS 特征。槽布局：lora[0]=HeadW；dense[0]=HeadB，
/// dense[1..]=QAM 扰动。可训练自由度 = head + QAM。
#[allow(clippy::too_many_arguments)]
fn es_forward_noconv(
    images: Tensor<B, 5>,
    base: &SdtWeights<B>,
    f: &ChunkFactors,
    cfg: &SdtConfig,
    relax_ssa: bool,
    relax_mlp: bool,
    beta: f32,
    qam: Option<&QamRun<B>>,
) -> Tensor<B, 2> {
    let t = images.dims()[0];
    let c = images.dims()[1];
    let lif = |x: Tensor<B, 5>, site: usize, th: f64| -> Tensor<B, 5> {
        match qam.and_then(|q| q.th5(site)) {
            Some(th_t) => lif_seq_bth(x, th_t, if relax_ssa { Some(beta as f64) } else { None }),
            None => {
                if relax_ssa {
                    lif_seq_relaxed(x, th, beta as f64)
                } else {
                    lif_seq(x, th)
                }
            }
        }
    };
    let lif_mlp = |x: Tensor<B, 5>, site: usize, th: f64| -> Tensor<B, 5> {
        match qam.and_then(|q| q.th5(site)) {
            Some(th_t) => lif_seq_bth(x, th_t, if relax_mlp { Some(beta as f64) } else { None }),
            None => {
                if relax_mlp {
                    lif_seq_relaxed(x, th, beta as f64)
                } else {
                    lif_seq(x, th)
                }
            }
        }
    };
    let lif4 = |x: Tensor<B, 4>, site: usize, th: f64| -> Tensor<B, 4> {
        match qam.and_then(|q| q.th4(site)) {
            Some(th_t) => lif_seq_bth(x, th_t, if relax_mlp { Some(beta as f64) } else { None }),
            None => {
                if relax_mlp {
                    lif_seq_relaxed(x, th, beta as f64)
                } else {
                    lif_seq(x, th)
                }
            }
        }
    };

    let mut x = forward_sps(images, base, cfg);
    let heads = cfg.num_heads;
    for _j in 0..base.blk.len() {
        let d = x.dims();
        let (_t, _b, ch, hh, ww) = (d[0], d[1], d[2], d[3], d[4]);
        let head_dim = ch / heads;

        // xs 位点 0；q/k/v 恒等（无 conv 投影）
        let xs = {
            let xin = match qam.and_then(|q| q.m(0)) {
                Some(m) => x.clone() * m.reshape([1, c, 256, 1, 1]),
                None => x.clone(),
            };
            lif(xin, 0, 1.0)
        };
        let identity = x;
        let qh = reshape_heads(xs.clone(), heads, head_dim);
        let kh = reshape_heads(xs.clone(), heads, head_dim);
        let vh = reshape_heads(xs, heads, head_dim);

        let kv = (kh * vh.clone()).sum_dim(3);
        let kv_spike = {
            let kin = match qam.and_then(|q| q.m(4)) {
                Some(m) => kv * m.reshape([1, c, heads, 1, head_dim]),
                None => kv,
            };
            lif(kin, 4, 0.5)
        };
        let xattn = qh * kv_spike;
        let xo = unshape_heads(xattn, heads, head_dim, hh, ww);
        let ssa_out = xo + identity; // proj 恒等

        // MLP：fc1/fc2 恒等
        let identity2 = ssa_out.clone();
        let x1 = {
            let xin = match qam.and_then(|q| q.m(5)) {
                Some(m) => ssa_out * m.reshape([1, c, 256, 1, 1]),
                None => ssa_out,
            };
            lif_mlp(xin, 5, 1.0)
        };
        let x2 = {
            let xin = match qam.and_then(|q| q.m(6)) {
                Some(m) => x1 * m.reshape([1, c, 256, 1, 1]),
                None => x1,
            };
            lif_mlp(xin, 6, 1.0)
        };
        x = x2 + identity2;
    }

    // head：与 conv 版相同（lora[0]=head 权重噪声，dense[0]=head 偏置噪声）
    let fd = x.dims();
    let (t, cb, ch) = (fd[0], fd[1], fd[2]);
    let feat = x
        .reshape([t, cb, ch, fd[3] * fd[4]])
        .sum_dim(3)
        .div_scalar((fd[3] * fd[4]) as f32);
    let feat = match qam.and_then(|q| q.m(7)) {
        Some(m) => feat * m.reshape([1, c, ch, 1]),
        None => feat,
    };
    let feat_spike = lif4(feat, 7, 1.0);
    let f3 = feat_spike.permute([1, 0, 2, 3]).reshape([c, t, ch]); // [C, T, 256]
    let (a_h, b_h) = &f.lora[0];
    let head_w2 = base.head_w.clone();
    let base_h = f3
        .clone()
        .reshape([cb * t, ch])
        .matmul(head_w2.transpose())
        .reshape([cb, t, cfg.num_classes]);
    let yn_h = f3.matmul(b_h.clone().swap_dims(1, 2)).matmul(a_h.clone());
    let hb = flatten_slot(&OrigShape::B1(cfg.num_classes), &base.head_b);
    let bias3 = (hb.reshape([1, cfg.num_classes]) + f.dense[0].clone()).reshape([c, 1, cfg.num_classes]);
    let logits3 = base_h + yn_h + bias3;
    logits3.sum_dim(1).squeeze::<2>().div_scalar(t as f32)
}

/// 纯线性探针前向（--blocks probe）：SPS → 时间/空间均值池化 → 线性 head。
/// 无 LIF、无块、无 QAM；槽布局：lora[0]=HeadW，dense[0]=HeadB。
/// 池化消去 H×W（ 与预训练 head 的输入语义一致：mean_pool @ W ）。
fn es_forward_probe(
    images: Tensor<B, 5>,
    base: &SdtWeights<B>,
    f: &ChunkFactors,
    cfg: &SdtConfig,
) -> Tensor<B, 2> {
    let t = images.dims()[0];
    let c = images.dims()[1];
    let x = forward_sps(images, base, cfg);
    let fd = x.dims();
    let (tt, cb, ch) = (fd[0], fd[1], fd[2]);
    let feat = x
        .reshape([tt, cb, ch, fd[3] * fd[4]])
        .sum_dim(3)
        .div_scalar((fd[3] * fd[4]) as f32); // [T, C, 256]
    let f3 = feat
        .permute([1, 0, 2, 3])
        .reshape([c, tt, ch]); // [C, T, 256]（feat 为 rank-4 [T,C,256,1]）
    let (a_h, b_h) = &f.lora[0];
    let head_w2 = base.head_w.clone();
    let base_h = f3
        .clone()
        .reshape([cb * tt, ch])
        .matmul(head_w2.transpose())
        .reshape([cb, tt, cfg.num_classes]);
    let yn_h = f3.matmul(b_h.clone().swap_dims(1, 2)).matmul(a_h.clone());
    let hb = flatten_slot(&OrigShape::B1(cfg.num_classes), &base.head_b);
    let bias3 = (hb.reshape([1, cfg.num_classes]) + f.dense[0].clone()).reshape([c, 1, cfg.num_classes]);
    let logits3 = base_h + yn_h + bias3;
    logits3.sum_dim(1).squeeze::<2>().div_scalar(tt as f32)
}

// ---------------------------------------------------------------------------
// 普通 SNN（--blocks fc）：HyperscaleES 主场——全 FC-LIF，无 SPS/无注意力。
// 展平像素[3072] → fc1[128]→LIF(0.3) → fc2[128]→LIF(0.3) → fc3[10] 时间平均读出。
// 基底随机初始化（N(0, 1/√fan_in)），ES 逐 epoch 累积低秩更新（从零训练语义）。
// ---------------------------------------------------------------------------

struct FcBase<B: Backend> {
    w1: Tensor<B, 2>, // [128, 3072]
    b1: Tensor<B, 1>, // [128]
    w2: Tensor<B, 2>, // [128, 128]
    b2: Tensor<B, 1>, // [128]
    w3: Tensor<B, 2>, // [10, 128]
    b3: Tensor<B, 1>, // [10]
}

fn gen_fc_base(seed: u64, device: &Dev) -> FcBase<B> {
    let mut rng = DeterministicNoise::new(seed);
    let (h, inp) = (128usize, 3072usize);
    let w1: Vec<f32> = (0..h * inp).map(|_| rng.standard_normal() * (inp as f32).sqrt().recip()).collect();
    let w2: Vec<f32> = (0..h * h).map(|_| rng.standard_normal() * (h as f32).sqrt().recip()).collect();
    let w3: Vec<f32> = (0..10 * h).map(|_| rng.standard_normal() * (h as f32).sqrt().recip()).collect();
    let b1 = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(vec![0.0_f32; h], [h]), device);
    let b2 = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(vec![0.0_f32; h], [h]), device);
    let b3 = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(vec![0.0_f32; 10], [10]), device);
    FcBase {
        w1: Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(w1, [h, inp]), device),
        b1,
        w2: Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(w2, [h, h]), device),
        b2,
        w3: Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(w3, [10, h]), device),
        b3,
    }
}

/// 普通 SNN 前向（噪声候选口径）：候选 [C]，时间 [T]，与 SDT 管线同 relax/β/QAM 机制
fn es_forward_fc(
    images: Tensor<B, 5>,
    base_fc: &FcBase<B>,
    f: &ChunkFactors,
    relax: bool,
    beta: f32,
    qam: Option<&QamRun<B>>,
) -> Tensor<B, 2> {
    let d = images.dims();
    let (t, c) = (d[0], d[1]);
    let x3 = images.reshape([t, c, 3072]).permute([1, 0, 2]); // [C,T,3072]

    let lif_fc = |cur: Tensor<B, 3>| -> Tensor<B, 3> {
        // [C,T,128] → LIF over T（v_th=0.3，与原始库一致；naive 1.0 会静默网络）
        let r5 = cur.permute([1, 0, 2]).reshape([t, c, 128, 1, 1]);
        let s = if relax {
            lif_seq_relaxed(r5, 0.3, beta as f64)
        } else {
            lif_seq(r5, 0.3)
        };
        s.reshape([t, c, 128]).permute([1, 0, 2])
    };

    // fc1
    let (a1, b1) = &f.lora[0];
    let base_cur = x3
        .clone()
        .reshape([c * t, 3072])
        .matmul(base_fc.w1.clone().transpose())
        .reshape([c, t, 128]);
    let yn1 = x3.matmul(b1.clone().swap_dims(1, 2)).matmul(a1.clone()); // [C,T,128]
    let mut cur1 = base_cur + yn1 + f.dense[0].clone().reshape([c, 1, 128]);
    if let Some(m) = qam.and_then(|q| q.m(0)) {
        cur1 = cur1 * m.reshape([c, 1, 128]);
    }
    let s1 = lif_fc(cur1);

    // fc2
    let (a2, b2) = &f.lora[1];
    let base_cur2 = s1
        .clone()
        .reshape([c * t, 128])
        .matmul(base_fc.w2.clone().transpose())
        .reshape([c, t, 128]);
    let yn2 = s1.matmul(b2.clone().swap_dims(1, 2)).matmul(a2.clone());
    let mut cur2 = base_cur2 + yn2 + f.dense[1].clone().reshape([c, 1, 128]);
    if let Some(m) = qam.and_then(|q| q.m(1)) {
        cur2 = cur2 * m.reshape([c, 1, 128]);
    }
    let s2 = lif_fc(cur2);

    // fc3 + 时间平均读出
    let (a3, b3) = &f.lora[2];
    let base_l = s2
        .clone()
        .reshape([c * t, 128])
        .matmul(base_fc.w3.clone().transpose())
        .reshape([c, t, 10]);
    let yn3 = s2.matmul(b3.clone().swap_dims(1, 2)).matmul(a3.clone());
    let logits3 = base_l + yn3 + f.dense[2].clone().reshape([c, 1, 10]);
    logits3.sum_dim(1).squeeze::<2>().div_scalar(t as f32)
}

/// 普通 SNN 参数扁平化（fc 槽位 → params2d）
fn factored_params2d_fc(base_fc: &FcBase<B>, lora: &[FSlotSpec], dense: &[FSlotSpec], device: &Dev) -> Vec<Tensor<B, 2>> {
    let mut out = Vec::new();
    for slot in lora {
        let t = match slot.loc {
            FLoc::FcW(k) => {
                let w = match k {
                    0 => &base_fc.w1,
                    1 => &base_fc.w2,
                    _ => &base_fc.w3,
                };
                flatten_slot(&slot.shape, w)
            }
            _ => unreachable!("fc lora 列表只含 FcW"),
        };
        out.push(t);
    }
    for slot in dense {
        let t = match slot.loc {
            FLoc::FcB(k) => {
                let b = match k {
                    0 => &base_fc.b1,
                    1 => &base_fc.b2,
                    _ => &base_fc.b3,
                };
                flatten_slot(&slot.shape, b)
            }
            FLoc::Qam(_, 0) => {
                let n = slot.shape.orig_dims()[0];
                Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(vec![0.1_f32; n], [n]), device)
                    .reshape([n, 1])
            }
            FLoc::Qam(_, 1) => {
                let n = slot.shape.orig_dims()[0];
                Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(vec![0.5_f32; n], [n]), device)
                    .reshape([n, 1])
            }
            _ => unreachable!("fc dense 列表只含 FcB/Qam"),
        };
        out.push(t);
    }
    out
}

/// 普通 SNN 写回
fn write_back_fc(base_fc: &mut FcBase<B>, lora: &[FSlotSpec], dense: &[FSlotSpec], params: &[Tensor<B, 2>]) {
    let nl = lora.len();
    for (i, slot) in lora.iter().enumerate() {
        let dims = slot.shape.orig_dims();
        let t: Tensor<B, 2> = params[i].clone().reshape([dims[0], dims[1]]);
        match slot.loc {
            FLoc::FcW(0) => base_fc.w1 = t,
            FLoc::FcW(1) => base_fc.w2 = t,
            FLoc::FcW(_) => base_fc.w3 = t,
            _ => unreachable!("fc lora 列表只含 FcW"),
        }
    }
    for (di, slot) in dense.iter().enumerate() {
        let i = nl + di;
        let n = slot.shape.orig_dims()[0];
        match slot.loc {
            FLoc::FcB(0) => base_fc.b1 = params[i].clone().reshape([n]),
            FLoc::FcB(1) => base_fc.b2 = params[i].clone().reshape([n]),
            FLoc::FcB(_) => base_fc.b3 = params[i].clone().reshape([n]),
            FLoc::Qam(_, _) => {}
            _ => unreachable!("fc dense 列表只含 FcB/Qam"),
        }
    }
}

/// 普通 SNN 验证（零噪声，评估口径 QamRun）
#[allow(clippy::too_many_arguments)]
fn eval_es_fc(
    base_fc: &FcBase<B>,
    lora_slots: &[FSlotSpec],
    dense_slots: &[FSlotSpec],
    rank: usize,
    qam_eval: Option<&QamWeights<B>>,
    data: &Cifar10Npz,
    batch: usize,
    t_steps: usize,
) -> f64 {
    let device = es_dev();
    let n = data.n_test;
    let order: Vec<usize> = (0..n).collect();
    let mut correct = 0usize;
    for chunk in order.chunks(batch) {
        let (images, targets) = data.get_batch(Split::Test, chunk, t_steps, device);
        let c = chunk.len();
        let factors = zero_factors(lora_slots, dense_slots, rank, c, device);
        let qam_run = qam_run_zero(c, qam_eval, device);
        let logits = es_forward_fc(images, base_fc, &factors, false, 4.0, qam_run.as_ref());
        let pred = logits.argmax(1).reshape([chunk.len()]);
        let pred_v: Vec<i64> = pred
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("读取预测失败");
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
        <B as burn::tensor::backend::Backend>::sync(device).expect("eval 逐批同步失败");
        <B as burn::tensor::backend::Backend>::memory_cleanup(device);
    }
    <B as burn::tensor::backend::Backend>::sync(device).expect("eval 同步失败");
    <B as burn::tensor::backend::Backend>::memory_cleanup(device);
    correct as f64 / n as f64 * 100.0
}

/// factored 主参数列表（lora 13 槽在前、dense 13 槽+QAM 槽在后，顺序与 factored_slots 一致）
fn factored_params2d(base: &SdtWeights<B>, lora: &[FSlotSpec], dense: &[FSlotSpec], device: &Dev) -> Vec<Tensor<B, 2>> {
    let mut out = Vec::new();
    for slot in lora {
        let t = match slot.loc {
            FLoc::BlkW(j, which) => flatten_slot(&slot.shape, floc_get4(base, j, which)),
            FLoc::HeadW => flatten_slot(&slot.shape, &base.head_w),
            FLoc::BlkB(_, _) | FLoc::HeadB | FLoc::Qam(_, _) | FLoc::Vth(_) | FLoc::FcW(_) | FLoc::FcB(_) => {
                unreachable!("lora 列表不应含 dense/QAM/Vth/Fc 槽")
            }
        };
        out.push(t);
    }
    for slot in dense {
        let t = match slot.loc {
            FLoc::BlkB(j, which) => flatten_slot(&slot.shape, floc_get1(base, j, which)),
            FLoc::HeadB => flatten_slot(&slot.shape, &base.head_b),
            // QAM 槽：无基底权重，恒等初始化 a=0.1, φ=0.5（m≈0.966，见 §12.5）
            FLoc::Qam(_, 0) => {
                let n = slot.shape.orig_dims()[0];
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::new(vec![0.1_f32; n], [n]),
                    device,
                )
                .reshape([n, 1])
            }
            FLoc::Qam(_, 1) => {
                let n = slot.shape.orig_dims()[0];
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::new(vec![0.5_f32; n], [n]),
                    device,
                )
                .reshape([n, 1])
            }
            // 可学习阈值：初始 = 硬编码默认（1.0，kv 位点 0.5）
            FLoc::Vth(site) => Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(
                    vec![if site == 4 { 0.5_f32 } else { 1.0_f32 }],
                    [1],
                ),
                device,
            )
            .reshape([1, 1]),
            _ => unreachable!(),
        };
        out.push(t);
    }
    out
}

/// 把 AdamW 更新后的 2D 主参数写回 base 权重集合
fn write_back_factored(base: &mut SdtWeights<B>, lora: &[FSlotSpec], dense: &[FSlotSpec], params: &[Tensor<B, 2>]) {
    let nl = lora.len();
    for (i, slot) in lora.iter().enumerate() {
        let dims = slot.shape.orig_dims();
        match slot.loc {
            FLoc::BlkW(j, which) => {
                let t: Tensor<B, 4> = params[i]
                    .clone()
                    .reshape([dims[0], dims[1], dims[2], dims[3]]);
                floc_set4(base, j, which, t);
            }
            FLoc::HeadW => {
                base.head_w = params[i].clone().reshape([dims[0], dims[1]]);
            }
            FLoc::BlkB(_, _) | FLoc::HeadB | FLoc::Qam(_, _) | FLoc::Vth(_) | FLoc::FcW(_) | FLoc::FcB(_) => {
                unreachable!("lora 列表不应含 dense/QAM/Vth/Fc 槽")
            }
        }
    }
    for (di, slot) in dense.iter().enumerate() {
        let i = nl + di;
        let n = slot.shape.orig_dims()[0];
        match slot.loc {
            FLoc::BlkB(j, which) => {
                let t: Tensor<B, 1> = params[i].clone().reshape([n]);
                floc_set1(base, j, which, t);
            }
            FLoc::HeadB => {
                base.head_b = params[i].clone().reshape([n]);
            }
            FLoc::BlkW(_, _) | FLoc::HeadW | FLoc::FcW(_) => unreachable!("dense 列表不应含 lora 槽"),
            // QAM/Vth/Fc 槽：状态就在 params2d 中（eval/mixed 时按槽列表重建），无写回
            FLoc::Qam(_, _) | FLoc::Vth(_) | FLoc::FcB(_) => {}
        }
    }
}

/// factored 模式主入口：逐图候选（噪声方向数 = pop），SPS 冻结，blocks/head 因式噪声
pub fn run_train_es_factored(args: EsArgs) {
    assert!(
        args.es_batch == 1,
        "factored 模式按逐图候选运行（es_batch 必须为 1，实际 {}）",
        args.es_batch
    );
    assert!(args.pop % 2 == 0, "pop 必须为偶数（反对称配对）");
    let cfg = SdtConfig::default();
    let device = es_dev();

    // ---- 数据 / 权重（与 cache 模式同源）----
    let mut data_path = format!("{}/cifar10_data.npz", args.data_dir.trim_end_matches(['/', '\\']));
    if !std::path::Path::new(&data_path).exists() {
        data_path = "artifacts/cifar10_data.npz".to_string();
    }
    println!("== Burn SDT 演化策略（EggRoll-ES，factored 因式噪声）训练 ==");
    println!(
        "数据: {data_path}（train={} test={}）T={} N_L(噪声方向)={} chunk={} σ={} rank={} lr={} (AdamW)",
        "8000", "2000", args.time_steps, args.pop, args.chunk, args.sigma, args.rank, args.lr
    );
    let data = Cifar10Npz::load(&data_path);
    assert!(args.pop <= data.n_train, "pop({}) 不能超过训练集({})", args.pop, data.n_train);

    let mut base: SdtWeights<B> = match &args.weights {
        Some(w) if !w.is_empty() => {
            println!("加载初始化权重: {w}");
            let npz = crate::tensor_io::read_npz(w);
            crate::model::load_weights(&npz, &cfg, device)
        }
        _ => panic!("ES 训练需要初始权重 NPZ（--weights）"),
    };
    if !args.no_calibrate {
        let t0 = std::time::Instant::now();
        base = crate::train::static_calibrate(&base, &data, device, &cfg);
        println!("[calibrate] 静态校准完成（耗时 {:.1}s）", t0.elapsed().as_secs_f32());
    }
    <B as burn::tensor::backend::Backend>::sync(device).expect("GPU 同步失败");
    <B as burn::tensor::backend::Backend>::memory_cleanup(device);

    // ---- 槽位与主参数 ----
    // QAM 位点（ES_MANIFOLD §12.3 位点深度律优先级）：head=feat 单位点（最高信号），
    // ssa=xs/q/k/v/kv，all=全部 8 位点；m 按位点类型跨 block 共享
    // probe 模式无 LIF，QAM 无处作用 → 强制关闭
    // fc 模式（普通 SNN）：LIF1/LIF2 输入电流调制，head=位点0，ssa/all=位点{0,1}
    let probe_mode = args.blocks == "probe";
    let fc_mode = args.blocks == "fc";
    if fc_mode && args.baseline_zero {
        panic!("普通 SNN（fc）模式暂不支持 --baseline-zero（锚点函数面向 SdtWeights）");
    }
    let qam_sites: Vec<usize> = if probe_mode {
        Vec::new()
    } else if fc_mode {
        match args.qam.as_str() {
            "off" => Vec::new(),
            "learnable" => match args.qam_sites.as_str() {
                "head" => vec![0],
                "ssa" | "all" => vec![0, 1],
                other => panic!("未知 --qam-sites: {other}（fc 模式可选 head|ssa|all）"),
            },
            other => panic!("未知 --qam: {other}（可选 off|learnable）"),
        }
    } else {
        match args.qam.as_str() {
            "off" => Vec::new(),
            "learnable" => match args.qam_sites.as_str() {
                "head" => vec![7],
                "ssa" => vec![0, 1, 2, 3, 4],
                "all" => vec![0, 1, 2, 3, 4, 5, 6, 7],
                other => panic!("未知 --qam-sites: {other}（可选 head|ssa|all）"),
            },
            other => panic!("未知 --qam: {other}（可选 off|learnable）"),
        }
    };
    let (lora_slots, dense_slots) = factored_slots(&base, cfg.depths, &qam_sites, &[], &args.blocks);
    // fc 模式：随机初始化基底（从零训练语义），--weights 仅用于 conv/none/probe
    let mut fc_base: Option<FcBase<B>> = if fc_mode {
        println!(
            "[普通SNN] 随机初始化 FC-LIF×2：3072→128→128→10，v_th=0.3（原始库口径），种子 {}",
            args.seed
        );
        Some(gen_fc_base(args.seed, device))
    } else {
        None
    };
    let mut params2d = if fc_mode {
        factored_params2d_fc(fc_base.as_ref().unwrap(), &lora_slots, &dense_slots, device)
    } else {
        factored_params2d(&base, &lora_slots, &dense_slots, device)
    };
    let qam_slot_base = lora_slots.len() + dense_slots.len() - 2 * qam_sites.len(); // 第一个 QAM 槽在 params2d 的下标
    let all_shapes: Vec<OrigShape> = lora_slots
        .iter()
        .chain(dense_slots.iter())
        .map(|s| s.shape.clone())
        .collect();
    let trainable: usize = all_shapes
        .iter()
        .map(|s| s.orig_dims().iter().product::<usize>())
        .sum();
    println!(
        "[es-factored] 可训练槽={}（LoRA={} dense={}）参数量={}（SPS 3x3 冻结）；SPS 冻结参数≈765k",
        all_shapes.len(), lora_slots.len(), dense_slots.len(), trainable
    );
    if !qam_sites.is_empty() {
        println!(
            "[es-factored] QAM 调制开启：mode=learnable m=(1+a)·cos(φ)，位点={:?}，槽参数={}（固定尺度 σ·c，c_a=0.1/c_φ=0.5）",
            qam_sites, trainable
        );
    }
    if args.relax {
        println!(
            "[es-factored] TSES 温度松弛开启：fitness 前向 LIF 读出 = σ(β(h−thr))，β₀={}，退火周期={}（×2，上限16）；验证仍用硬脉冲 forward_full",
            args.beta, args.beta_anneal_every
        );
    }
    let mut optim = AdamW::new(&all_shapes, args.lr as f32, 1e-4, device);

    // S2 优化：训练集一次性预载显存（8000×3072×4B ≈ 98MB），chunk 批次改为
    // GPU 端 one-hot matmul 行选择（0/1 矩阵乘在 f32 下逐位精确 = 行拷贝，
    // ~6GFLOP/chunk 可忽略），删除 62 次/epoch 的 CPU gather + 图像 PCIe 上传。
    let frame = crate::loader::C * crate::loader::H * crate::loader::W;
    let pix_t = Tensor::<B, 1>::from_floats(data.pixels_flat(Split::Train), device)
        .reshape([data.n_train, frame]); // [N, 3072]
    let lab_t = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(
            data.labels_i64(Split::Train).to_vec(),
            [data.n_train],
        ),
        device,
    );
    let ar_t = Tensor::<B, 1, Int>::arange(0..data.n_train as i64, device)
        .reshape([1, data.n_train]);

    // CSV
    let mut csv_rows: Vec<String> =
        vec!["epoch,train_top1,train_loglik,val_top1,best_val,epoch_time,cum_time".to_string()];
    let mut best_val = 0.0_f64;
    let mut best_train = 0.0_f64;
    let mut cum_t = 0.0_f64;

    // X3 逐槽步长自适应状态（NES/CSA 坐标级 σ 自适应的漂移幅值版）
    let n_slots_total = lora_slots.len() + dense_slots.len();
    let mut ema_g: Vec<f32> = vec![0.0; n_slots_total];
    let mut sigma_scale: Vec<f32> = vec![1.0; n_slots_total];

    for epoch in 0..args.epochs {
        let ep_start = std::time::Instant::now();
        let order_full = crate::loader::shuffled_indices(data.n_train, args.seed + epoch as u64);
        let order = &order_full[..args.pop]; // 候选 = 图（逐图噪声方向）
        // TSES β 调度：每 beta_anneal_every epoch ×2，上限 16
        let beta_t = if args.relax && args.beta_anneal_every > 0 {
            (args.beta * 2f32.powi((epoch / args.beta_anneal_every) as i32)).min(16.0)
        } else {
            args.beta
        };

        // 可变学习率：线性 warmup + 余弦退火（lr_warmup>0 启用；ES 每 epoch 一次 AdamW step）
        let lr_t = if args.lr_warmup > 0 {
            let ep = epoch as f64;
            let w = args.lr_warmup as f64;
            let tot = args.epochs as f64;
            if epoch < args.lr_warmup {
                args.lr * ((ep + 1.0) / w)
            } else {
                let prog = ((ep - w) / (tot - w).max(1.0)).min(1.0);
                let cosv = 0.5 * (1.0 + (std::f64::consts::PI * prog).cos());
                args.lr * (args.lr_min_frac + (1.0 - args.lr_min_frac) * cosv)
            }
        } else {
            args.lr
        };
        optim.set_lr(lr_t as f32);

        // σ_slot 每 epoch 重算（随参数演化自适应）
        // QAM 槽用固定尺度（初始近常数 std≈0，std 口径会退化为 0）：σ·c，c_a=0.1/c_φ=0.5
        // freeze_lora：块卷积权重槽（BlkW = lora[0..nl-1]）σ=0（head 槽保留）；
        // none/probe 模式 lora 只有 head 槽 → freeze 无对象
        let nl_conv = if args.blocks == "conv" { lora_slots.len() - 1 } else { 0 };
        let mut sigma_lora = Vec::with_capacity(lora_slots.len());
        let mut sigma_dense = Vec::with_capacity(dense_slots.len());
        for (i, p) in params2d.iter().enumerate() {
            let n = p.dims()[0] * p.dims()[1];
            let mean = p.clone().sum().into_scalar() / n as f32;
            let var = (p.clone().powf_scalar(2.0).sum().into_scalar() / n as f32 - mean * mean).max(0.0);
            let s = if i >= qam_slot_base {
                let which = (i - qam_slot_base) % 2;
                args.sigma * if which == 0 { 0.1 } else { 0.5 }
            } else if args.freeze_lora && i < nl_conv {
                0.0
            } else {
                args.sigma * var.sqrt()
            };
            // X3：逐槽自适应比例（每槽独立缩放，含 QAM 槽）
            let s = s * sigma_scale[i];
            if i < lora_slots.len() {
                sigma_lora.push(s);
            } else {
                sigma_dense.push(s);
            }
        }

        let mut grad_acc: Vec<Tensor<B, 2>> = params2d
            .iter()
            .map(|p| Tensor::<B, 2>::zeros(p.dims(), device))
            .collect();
        // S1 优化：raw 统计与正确数全程 GPU 累积，epoch 末只读 3 个标量
        // （删除 62×2 次 into_scalar 强制同步——每次 sync 都会打断 CPU↔GPU 流水线）
        let mut raw_sum_t = Tensor::<B, 1>::zeros([1], device);
        let mut raw_sumsq_t = Tensor::<B, 1>::zeros([1], device);
        let mut correct_acc = Tensor::<B, 1>::zeros([1], device);
        let mut n_used = 0usize;

        let mut gen_time = 0.0_f32;
        let mut fwd_time = 0.0_f32;
        let n_chunks = args.pop / args.chunk;
        let order_vec: Vec<usize> = order.to_vec();
        // X1 控制变量锚点：每 epoch 一次无噪声前向（当前 θ + 当前 QAM 基值），
        // 逐训练图 loglik b₀(x)。relax 口径与候选 fitness 一致（同 relax/β）。
        let anchors_gpu: Option<Tensor<B, 1>> = if args.baseline_zero {
            let a_relax_fwd = args.relax && (beta_t < 16.0 || !args.hard_at_16);
            let a_relax_mlp = a_relax_fwd && args.relax_scope != "ssa";
            // 当前 QAM 基值（与验证节同构，从 params2d 重建）
            let qam_eval_anchor = if qam_sites.is_empty() {
                None
            } else {
                let mut sites = Vec::with_capacity(qam_sites.len());
                for (kk, &site) in qam_sites.iter().enumerate() {
                    let ia = qam_slot_base + 2 * kk;
                    let na = params2d[ia].dims()[0];
                    sites.push((
                        site,
                        params2d[ia].clone().reshape([na]),
                        params2d[ia + 1].clone().reshape([na]),
                    ));
                }
                Some(QamWeights { sites, vth: Vec::new() })
            };
            let t_a = std::time::Instant::now();
            let anchors = zero_noise_anchors(
                &base, &lora_slots, &dense_slots, qam_eval_anchor.as_ref(), &data, &cfg,
                &args.blocks, args.rank, &order_vec, 512, args.time_steps,
                a_relax_fwd, a_relax_mlp, beta_t,
            );
            println!(
                "  [anchor] b₀ mean={:.4} std={:.4}（{:.1}s）",
                anchors.iter().sum::<f32>() / anchors.len() as f32,
                {
                    let m = anchors.iter().sum::<f32>() / anchors.len() as f32;
                    (anchors.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / anchors.len() as f32).sqrt()
                },
                t_a.elapsed().as_secs_f32()
            );
            let n_anchor = anchors.len();
            Some(Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(anchors, [n_anchor]),
                device,
            ))
        } else {
            None
        };
        for k in 0..n_chunks {
            // ---- 1) S3：GPU 端因子生成（零 CPU 生成、零 PCIe 上传）----
            let t0 = std::time::Instant::now();
            // --pair-shared：候选 2k/2k+1 共享同一样本（位置 p 的图像 = order[p>>1]），
            // 噪声 id/符号按全局位置（与原始 eggroll 的 thread_id 语义一致：pair=id/2，符号=id 奇偶）。
            // 默认（false）保持历史行为：id=图像行号，± 对各自评估不同样本（数据项混入 z 差分）。
            let (ids_slice, img_slice): (Vec<usize>, Vec<usize>) = if args.pair_shared {
                let pos: Vec<usize> = (k * args.chunk..(k + 1) * args.chunk).collect();
                let imgs = pos.iter().map(|&p| order_vec[p >> 1]).collect();
                (pos, imgs)
            } else {
                let c = order_vec[k * args.chunk..(k + 1) * args.chunk].to_vec();
                let d = c.clone();
                (c, d)
            };
            let cand_slice = ids_slice;
            let factors = gen_chunk_factors_gpu(
                &lora_slots,
                &dense_slots,
                &sigma_lora,
                &sigma_dense,
                args.rank,
                &cand_slice,
                epoch as i32,
                device,
            );
            gen_time += t0.elapsed().as_secs_f32();
            // QAM 运行时：基底参数（params2d）+ 逐候选噪声（factors.dense 末段）
            let qam_run = if qam_sites.is_empty() {
                None
            } else {
                let nd_total = dense_slots.len();
                let mut sites = Vec::with_capacity(qam_sites.len());
                for (kk, &site) in qam_sites.iter().enumerate() {
                    let ia = qam_slot_base + 2 * kk;
                    let di = nd_total - 2 * qam_sites.len() + 2 * kk;
                    sites.push((
                        site,
                        params2d[ia].clone(),
                        params2d[ia + 1].clone(),
                        factors.dense[di].clone(),
                        factors.dense[di + 1].clone(),
                    ));
                }
                Some(QamRun { sites, vth: Vec::new() })
            };

            // ---- 2) 因式噪声前向 + fitness ----
            let t1 = std::time::Instant::now();
            // S2：GPU 行选择替代 get_batch（语义与 [T,B,3,32,32] 逐位一致）
            // 图像/标签按 img_slice（--pair-shared 时相邻两位同图）
            let idx_t = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(
                    img_slice.iter().map(|&i| i as i64).collect::<Vec<i64>>(),
                    [img_slice.len()],
                ),
                device,
            );
            let onehot = idx_t.clone().reshape([img_slice.len(), 1]).equal(ar_t.clone()).float(); // [C,N]
            let sel = onehot.matmul(pix_t.clone()); // [C, 3072]（f32 精确行拷贝）
            let images = sel
                .reshape([img_slice.len(), 3, 32, 32])
                .reshape([1, img_slice.len(), 3, 32, 32])
                .repeat_dim(0, args.time_steps); // [T, C, 3, 32, 32]
            let targets = lab_t.clone().select(0, idx_t.clone()); // [C] Int
            // S4 优化：β 退火到 16 后切硬 LIF——Run-1（1000ep）实测为训练质量负优化
            // （峰值后衰减回归，机制 = 定理 2/3：硬 fitness 信号只在切换复形上），
            // 故改为 opt-in：默认全程松弛（v4 行为），--hard-at-16 启用硬切换。
            // S6：relax-scope=ssa 时 MLP/head 位点保持硬 LIF
            let relax_fwd = args.relax && (beta_t < 16.0 || !args.hard_at_16);
            let relax_mlp = relax_fwd && args.relax_scope != "ssa";
            let logits = match args.blocks.as_str() {
                "conv" => es_forward_factored(images, &base, &factors, &cfg, relax_fwd, relax_mlp, beta_t, qam_run.as_ref()), // [C, 10]
                "none" => es_forward_noconv(images, &base, &factors, &cfg, relax_fwd, relax_mlp, beta_t, qam_run.as_ref()),
                "probe" => es_forward_probe(images, &base, &factors, &cfg),
                "fc" => es_forward_fc(images, fc_base.as_ref().unwrap(), &factors, relax_fwd, beta_t, qam_run.as_ref()),
                other => panic!("未知 --blocks: {other}"),
            }; // [C, 10]
            let logsm = log_softmax2(logits.clone());
            // S7 优化：one_hot 构造+乘法 → gather（raw[c] = logsm[c, targets[c]]，逐位一致）
            let mut raw = logsm
                .gather(1, targets.clone().reshape([img_slice.len(), 1]))
                .reshape([img_slice.len()]); // [C]
            // X1 控制变量：raw′ₙ = rawₙ − b₀(xₙ)（逐候选零噪声锚点）
            if let Some(an) = anchors_gpu.as_ref() {
                let an_g = an.clone().select(0, idx_t.clone());
                raw = raw - an_g;
            }
            let pred = logits.argmax(1).reshape([cand_slice.len()]);
            correct_acc = correct_acc.clone() + pred.equal(targets).float().sum();
            n_used += cand_slice.len();
            fwd_time += t1.elapsed().as_secs_f32();

            // ---- 3) 梯度充分统计量累积（同一批因子，无 CPU 回传）----
            // M1 优化（ES_MANIFOLD_NOTE.md §一）：反对称对共享 seed、ΔW 严格互为相反数
            // （gen_chunk_flat：A 区乘 ±sign、B 区共享；dense 整体 ±），
            // pop=n_train 时每对完整 ⇒ Σ_n ΔW_n ≡ 0（精确恒零）⇒ ones_acc 全删。
            for si in 0..lora_slots.len() {
                let (a, b) = &factors.lora[si];
                grad_acc[si] = grad_acc[si].clone() + lora_grad_einsum(a, b, &raw, true);
            }
            let nd = lora_slots.len();
            for di in 0..dense_slots.len() {
                let noise = &factors.dense[di];
                let g = noise.clone().transpose().matmul(raw.clone().reshape([cand_slice.len(), 1])); // [n,1]
                grad_acc[nd + di] = grad_acc[nd + di].clone() + g;
            }
            raw_sum_t = raw_sum_t.clone() + raw.clone().sum();
            raw_sumsq_t = raw_sumsq_t.clone() + raw.powf_scalar(2.0).sum();
        }

        // ---- 4) 全局 z-score（一次）+ 单次 AdamW + 写回 ----
        // pop=n_train 时 ΣΔW≡0（M1），修正项 (g − o·mean) ≡ g，直接乘 scale。
        // S1：epoch 末一次性回读 3 个标量（raw_sum/raw_sumsq/correct）
        let n_used = n_used; // 已在循环内累加
        let s1v = raw_sum_t.into_scalar();
        let s2v = raw_sumsq_t.into_scalar();
        let ccv = correct_acc.into_scalar();
        let mean = s1v / n_used as f32;
        let var = (s2v / n_used as f32 - mean * mean).max(0.0);
        let stdv = (var + 1e-5).sqrt();
        let scale = -1.0 / (stdv * (n_used as f32).sqrt());
        let grads: Vec<Tensor<B, 2>> = grad_acc
            .into_iter()
            .map(|g| g.mul_scalar(scale))
            .collect();
        // X4 漂移范数：‖g‖（GMD 平衡点监控——收敛时归零）
        let gnorm2: f32 = grads
            .iter()
            .map(|g| g.clone().powf_scalar(2.0).sum().into_scalar())
            .sum();
        let gnorm = gnorm2.sqrt();
        // X3：逐槽漂移幅值 EMA + 每 10 epoch 一次 σ 比例自适应
        if args.sigma_adapt {
            for (i, g) in grads.iter().enumerate() {
                let n = g.clone().powf_scalar(2.0).sum().into_scalar().sqrt();
                ema_g[i] = if epoch == 0 { n } else { 0.7 * ema_g[i] + 0.3 * n };
            }
            if (epoch + 1) % args.sigma_adapt_every.max(1) == 0 {
                let mut sortedv = ema_g.clone();
                sortedv.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = sortedv[sortedv.len() / 2].max(1e-12);
                for (i, e) in ema_g.iter().enumerate() {
                    let r = (e + 1e-12) / med;
                    sigma_scale[i] = (sigma_scale[i] * r.sqrt()).clamp(0.25, 4.0);
                }
                let slot_name = |k: usize| -> String {
                    if k < lora_slots.len() {
                        format!("{:?}", lora_slots[k].loc)
                    } else {
                        format!("{:?}", dense_slots[k - lora_slots.len()].loc)
                    }
                };
                let mut idx: Vec<usize> = (0..n_slots_total).collect();
                idx.sort_by(|&a, &b| sigma_scale[b].partial_cmp(&sigma_scale[a]).unwrap());
                println!(
                    "  [σ-adapt] ↑{} {:.2}× ↓{} {:.2}×（epoch {}）",
                    slot_name(idx[0]),
                    sigma_scale[idx[0]],
                    slot_name(idx[n_slots_total - 1]),
                    sigma_scale[idx[n_slots_total - 1]],
                    epoch + 1
                );
                for e in ema_g.iter_mut() {
                    *e = med;
                }
            }
        }
        // freeze_lora：AdamW 的 wd 会衰减零梯度槽——步前保存、步后恢复，保证严格冻结
        let frozen_saved: Option<Vec<Tensor<B, 2>>> = if args.freeze_lora && nl_conv > 0 {
            Some(params2d[..nl_conv].to_vec())
        } else {
            None
        };
        optim.step(&mut params2d, &grads);
        if let Some(saved) = frozen_saved {
            for (i, s) in saved.into_iter().enumerate() {
                params2d[i] = s;
            }
        }
        if fc_mode {
            write_back_fc(fc_base.as_mut().unwrap(), &lora_slots, &dense_slots, &params2d);
        } else {
            write_back_factored(&mut base, &lora_slots, &dense_slots, &params2d);
        }

        // ---- 5) epoch 末 sync + cleanup ----
        <B as burn::tensor::backend::Backend>::sync(device).expect("GPU 同步失败");
        <B as burn::tensor::backend::Backend>::memory_cleanup(device);

        let train_top1 = ccv as f64 / n_used as f64 * 100.0;
        let train_loglik = mean as f64;
        if train_top1 > best_train {
            best_train = train_top1;
        }

        // ---- 6) 验证 ----
        let mut val_str = String::new();
        if epoch % args.validate_every == 0 || epoch == args.epochs - 1 {
            // QAM 评估权重：从 params2d 的 QAM 槽重建（无噪声、硬前向）
            let qam_eval = if qam_sites.is_empty() {
                None
            } else {
                let mut sites = Vec::with_capacity(qam_sites.len());
                for (kk, &site) in qam_sites.iter().enumerate() {
                    let ia = qam_slot_base + 2 * kk;
                    let na = params2d[ia].dims()[0];
                    sites.push((
                        site,
                        params2d[ia].clone().reshape([na]),
                        params2d[ia + 1].clone().reshape([na]),
                    ));
                }
                Some(QamWeights { sites, vth: Vec::new() })
            };
            let val_top1 = if fc_mode {
                eval_es_fc(
                    fc_base.as_ref().unwrap(),
                    &lora_slots,
                    &dense_slots,
                    args.rank,
                    qam_eval.as_ref(),
                    &data,
                    256,
                    args.time_steps,
                )
            } else {
                eval_es_arch(&base, qam_eval.as_ref(), &data, 256, args.time_steps, &cfg, &args.blocks) // S9：批 64→256
            };
            if val_top1 > best_val {
                best_val = val_top1;
            }
            val_str = format!("{val_top1:.2}");
            println!(
                "epoch={}/{}, train_top1={:.2}%, train_loglik={:.4}, val_top1={:.2}%, best_val={:.2}%{}（gen {:.1}s fwd {:.1}s，本轮 {:.1}s，lr={:.5}，‖g‖={:.3}）",
                epoch + 1, args.epochs, train_top1, train_loglik, val_top1, best_val,
                if args.relax {
                    if args.relax && (beta_t < 16.0 || !args.hard_at_16) {
                        format!(" β={beta_t:.1}(松弛前向)")
                    } else {
                        format!(" β={beta_t:.1}(硬前向)")
                    }
                } else {
                    String::new()
                },
                gen_time, fwd_time,
                ep_start.elapsed().as_secs_f32(),
                lr_t,
                gnorm
            );
        } else {
            println!(
                "epoch={}/{}, train_top1={:.2}%, train_loglik={:.4}（gen {:.1}s fwd {:.1}s，本轮 {:.1}s）",
                epoch + 1, args.epochs, train_top1, train_loglik, gen_time, fwd_time,
                ep_start.elapsed().as_secs_f32()
            );
        }

        let el = ep_start.elapsed().as_secs_f64();
        cum_t += el;
        csv_rows.push(format!(
            "{},{:.2},{:.6},{},{:.2},{:.1},{:.1}",
            epoch + 1, train_top1, train_loglik, val_str, best_val, el, cum_t
        ));
        write_csv(&args.csv_out, &csv_rows);
    }

    println!("ES(factored) 训练完成：best_val={best_val:.2}% best_train={best_train:.2}% 总耗时 {cum_t:.0}s");
    println!("训练指标已写入: {}", args.csv_out);
}

// ---------------------------------------------------------------------------
// 混合估计器（ES_MANIFOLD_NOTE §13）：
//   SGD（精确梯度，momentum 0.9）管理平滑权重 W/b（含 SPS）——无 N/D 限制；
//   ES 只管理低维对偶自由度：可学习阈值 v_th（8 标量，全位点）+ QAM (a, φ) 槽
//   （对偶域的 θ_i/α_i 坐标）。ES fitness = 松弛前向（β 退火）逐候选 loglik，
//   σ 槽固定尺度（Run-4 证伪 lr 耦合衰减），每 epoch 一次 z-score + AdamW(wd=0)。
//   v_th 以逐候选阈值张量进入 fitness 前向（lif_seq_bth），验证/部署 = 硬前向 + 当前 v_th。
// ---------------------------------------------------------------------------

/// 混合估计器参数
pub struct MixedArgs {
    pub epochs: u32,
    pub data_dir: String,
    pub seed: u64,
    pub weights: Option<String>,
    pub time_steps: usize,
    pub no_calibrate: bool,
    /// SGD 批大小（子集 8000，默认 128 → 62 步/epoch）
    pub batch_size: usize,
    /// SGD 学习率（cosine 到 lr_sgd_min_frac×）
    pub lr_sgd: f64,
    pub lr_sgd_min_frac: f64,
    /// SGD 动量（与 train.rs 基线一致 0.9）
    pub sgd_momentum: f64,
    /// ES 候选数（= 训练集大小，反对称对完整）
    pub es_pop: usize,
    pub es_chunk: usize,
    /// ES 扰动基准尺度（QAM c_a=0.1/c_φ=0.5；v_th c=0.2）
    pub sigma_es: f32,
    /// ES 更新学习率（AdamW，wd=0——零梯度槽不衰减）
    pub lr_es: f32,
    pub beta: f32,
    pub beta_anneal_every: u32,
    /// QAM："off" | "learnable"
    pub qam: String,
    /// QAM 位点："head" | "ssa" | "all"
    pub qam_sites: String,
    pub validate_every: u32,
    pub csv_out: String,
}

/// 混合估计器主入口：每 epoch = 1 个 SGD epoch（W/b 全参数）+ 1 次 ES 低维更新
pub fn run_train_mixed(args: MixedArgs) {
    let cfg = SdtConfig::default();
    let device = es_dev();
    println!(
        "== 混合估计器：SGD(W/b, momentum {}) + ES(v_th 8 标量 + QAM) ==",
        args.sgd_momentum
    );

    // 数据（与 ES 线同一子集，pop=n_train；回退链与 train.rs 一致）
    let mut data_path = format!("{}/cifar10_data.npz", args.data_dir.trim_end_matches(['/', '\\']));
    if !std::path::Path::new(&data_path).exists() {
        data_path = "artifacts/cifar10_data.npz".to_string();
    }
    println!("数据: {data_path}（train={} test={}）", "8000", "2000");
    let data = Cifar10Npz::load(&data_path);

    // 权重（ES 线同款：校准后 init NPZ）
    let mut base: SdtWeights<BackendAdapter> = match &args.weights {
        Some(w) if !w.is_empty() => {
            println!("加载初始化权重: {w}");
            let npz = crate::tensor_io::read_npz(w);
            crate::model::load_weights(&npz, &cfg, device)
        }
        _ => panic!("混合估计器需要初始权重 NPZ（--weights）"),
    };
    if !args.no_calibrate {
        let t0 = std::time::Instant::now();
        base = crate::train::static_calibrate(&base, &data, device, &cfg);
        println!("[calibrate] 静态校准完成（耗时 {:.1}s）", t0.elapsed().as_secs_f32());
    }
    <B as burn::tensor::backend::Backend>::sync(device).expect("GPU 同步失败");
    <B as burn::tensor::backend::Backend>::memory_cleanup(device);

    // ---- ES 侧槽位（v_th 全 8 位点 + QAM）----
    let qam_sites: Vec<usize> = match args.qam.as_str() {
        "off" => Vec::new(),
        "learnable" => match args.qam_sites.as_str() {
            "head" => vec![7],
            "ssa" => vec![0, 1, 2, 3, 4],
            "all" => vec![0, 1, 2, 3, 4, 5, 6, 7],
            other => panic!("未知 --qam-sites: {other}"),
        },
        other => panic!("未知 --qam: {other}"),
    };
    let vth_sites: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7]; // 全位点标量阈值
    let (lora_slots, dense_slots) = factored_slots(&base, cfg.depths, &qam_sites, &vth_sites, "conv");
    let mut es_params: Vec<Tensor<B, 2>> =
        factored_params2d(&base, &lora_slots, &dense_slots, device);
    let nd_bias = dense_slots.len() - qam_sites.len() * 2 - vth_sites.len(); // 前 13 个 = 真实偏置槽
    let es_slot_base = lora_slots.len() + nd_bias; // 第一个 ES 槽（QAM/Vth）下标
    let n_es_slots = dense_slots.len() - nd_bias;
    println!(
        "[mixed] ES 槽：QAM={}×2 + v_th={}，自由度={}（SGD 侧 ~1.58M 平滑权重）",
        qam_sites.len(),
        vth_sites.len(),
        n_es_slots
    );

    // σ 布局：lora 全 0（W/b 走 SGD）；偏置槽全 0；QAM 固定尺度；v_th 固定 0.2σ
    let sigma_lora_es = vec![0.0_f32; lora_slots.len()];
    let mut sigma_dense_es = vec![0.0_f32; nd_bias];
    for (k, _) in qam_sites.iter().enumerate() {
        sigma_dense_es.push(args.sigma_es * 0.1); // c_a
        sigma_dense_es.push(args.sigma_es * 0.5); // c_φ
    }
    for _ in &vth_sites {
        sigma_dense_es.push(args.sigma_es * 0.2);
    }
    debug_assert_eq!(sigma_dense_es.len(), dense_slots.len());

    // ES 优化器（wd=0：零梯度槽不衰减）；m/v 必须覆盖全部槽（lora+dense，与
    // es_params/grads 的 1:1 下标对齐——否则 AdamW step 内形状错位）
    let es_shapes: Vec<OrigShape> = lora_slots
        .iter()
        .chain(dense_slots.iter())
        .map(|s| s.shape.clone())
        .collect();
    let mut optim_es = AdamW::new(&es_shapes, args.lr_es, 0.0, device);

    // ---- SGD 侧（autodiff 模型 + momentum SGD）----
    let ad_device: AutodiffDevice = Default::default();
    let init_ad = to_autodiff_weights(&base);
    let mut model = TrainSdt::<AutodiffBackend>::from_weights(&init_ad);
    println!("[mixed] SGD 可训练参数量: {}", model.num_params());
    let mut optim_sgd = burn::optim::SgdConfig::new()
        .with_momentum(Some(burn::optim::momentum::MomentumConfig {
            momentum: args.sgd_momentum,
            dampening: 0.0,
            nesterov: false,
        }))
        .init::<AutodiffBackend, TrainSdt<AutodiffBackend>>();

    // v_th 当前值（CPU 标量，从 es_params 读出）
    let mut vth_cpu: Vec<(usize, f64)> = vth_sites
        .iter()
        .enumerate()
        .map(|(k, &site)| {
            let idx = es_slot_base + qam_sites.len() * 2 + k;
            (site, es_params[idx].clone().into_scalar() as f64)
        })
        .collect();
    println!(
        "[mixed] v_th 初始: {:?}",
        vth_cpu.iter().map(|(s, v)| format!("{s}:{v:.2}")).collect::<Vec<_>>()
    );

    let t = args.time_steps;
    let mut csv_rows: Vec<String> =
        vec!["epoch,sgd_loss,es_loglik,val_top1,best_val,epoch_time,cum_time".to_string()];
    let mut best_val = 0.0_f64;
    let mut cum_t = 0.0_f64;

    for epoch in 0..args.epochs {
        let ep_start = std::time::Instant::now();
        // β 退火（与 ES 线一致：每 N epoch ×2，上限 16）
        let beta_t = if args.beta_anneal_every > 0 {
            (args.beta * 2f32.powi((epoch / args.beta_anneal_every) as i32)).min(16.0)
        } else {
            args.beta
        };

        // ---- 1) SGD epoch（W/b 精确梯度；前向带当前 QAM m 与 v_th）----
        // QAM m 的 SGD 视图：a/phi 从 es_params 转入 autodiff 后端（[n,1]→[n]）
        let qam_sgd = if qam_sites.is_empty() {
            None
        } else {
            let mut sites = Vec::with_capacity(qam_sites.len());
            for (k, &site) in qam_sites.iter().enumerate() {
                let ia = es_slot_base + 2 * k;
                let na = es_params[ia].dims()[0];
                sites.push((
                    site,
                    burn::tensor::Tensor::<AutodiffBackend, 1>::from_inner(
                        es_params[ia].clone().reshape([na]),
                    ),
                    burn::tensor::Tensor::<AutodiffBackend, 1>::from_inner(
                        es_params[ia + 1].clone().reshape([na]),
                    ),
                ));
            }
            Some(QamWeights { sites, vth: vth_cpu.clone() })
        };
        let lr_sgd_t = {
            let ep = epoch as f64;
            let tot = args.epochs as f64;
            let prog = (ep / tot.max(1.0)).min(1.0);
            let cosv = 0.5 * (1.0 + (std::f64::consts::PI * prog).cos());
            args.lr_sgd * (args.lr_sgd_min_frac + (1.0 - args.lr_sgd_min_frac) * cosv)
        };
        let order = shuffled_indices(data.n_train, args.seed + epoch as u64);
        let (sgd_loss, _) = crate::train::train_epoch(
            &mut model,
            &mut optim_sgd,
            &data,
            &order,
            args.batch_size,
            t,
            lr_sgd_t,
            &ad_device,
            &cfg,
            qam_sgd.as_ref(),
        );

        // ---- 2) ES 低维更新（fitness 前向用 SGD 后的 W/b）----
        let weights_inner = to_wgpu_weights(&model.to_weights());
        let vth_base: Vec<(usize, f64)> = vth_cpu.clone();
        let mut grad_acc: Vec<Tensor<B, 2>> = es_params
            .iter()
            .map(|p| Tensor::<B, 2>::zeros(p.dims(), device))
            .collect();
        let mut raw_sum_t = Tensor::<B, 1>::zeros([1], device);
        let mut raw_sumsq_t = Tensor::<B, 1>::zeros([1], device);
        let mut n_used = 0usize;
        let mut es_loglik = 0.0_f64;

        // 候选张量缓存（S2 同款行选择）
        let n_train = data.n_train;
        let frame = crate::loader::C * crate::loader::H * crate::loader::W;
        let pix_t = Tensor::<B, 1>::from_floats(data.pixels_flat(Split::Train), device)
            .reshape([n_train, frame]); // [N, 3072]
        let lab_t = Tensor::<B, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(
                data.labels_i64(Split::Train).to_vec(),
                [n_train],
            ),
            device,
        );
        let ar_t = Tensor::<B, 1, burn::tensor::Int>::arange(0..n_train as i64, device)
            .reshape([1, n_train]);
        let order_vec: Vec<usize> = (0..args.es_pop).collect();

        let n_chunks = args.es_pop / args.es_chunk;
        for k in 0..n_chunks {
            let cand_slice: Vec<usize> = order_vec[k * args.es_chunk..(k + 1) * args.es_chunk].to_vec();
            let factors = gen_chunk_factors_gpu(
                &lora_slots,
                &dense_slots,
                &sigma_lora_es,
                &sigma_dense_es,
                4, // lora 槽 σ=0，rank 只影响被丢弃的零噪声缓冲大小
                &cand_slice,
                epoch as i32,
                device,
            );
            // QamRun：QAM 扰动 + v_th 扰动
            let qam_run = {
                let mut sites = Vec::new();
                for (kk, &site) in qam_sites.iter().enumerate() {
                    let ia = es_slot_base + 2 * kk;
                    let di = nd_bias + 2 * kk;
                    sites.push((
                        site,
                        es_params[ia].clone(),
                        es_params[ia + 1].clone(),
                        factors.dense[di].clone(),
                        factors.dense[di + 1].clone(),
                    ));
                }
                let mut vthr = Vec::new();
                for (kk, &site) in vth_sites.iter().enumerate() {
                    let idx = es_slot_base + qam_sites.len() * 2 + kk;
                    let di = nd_bias + qam_sites.len() * 2 + kk;
                    let (_s, base_v) = vth_base.iter().find(|(s, _)| *s == site).unwrap();
                    vthr.push((site, *base_v, factors.dense[di].clone().reshape([cand_slice.len()])));
                }
                QamRun { sites, vth: vthr }
            };
            // 行选择 + 前向（松弛 fitness，β 退火；SSA+MLP 全松弛）
            let idx_t = Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(
                    cand_slice.iter().map(|&i| i as i64).collect::<Vec<i64>>(),
                    [cand_slice.len()],
                ),
                device,
            );
            let onehot = idx_t.clone().reshape([cand_slice.len(), 1]).equal(ar_t.clone()).float();
            let sel = onehot.matmul(pix_t.clone());
            let images = sel
                .reshape([cand_slice.len(), 3, 32, 32])
                .reshape([1, cand_slice.len(), 3, 32, 32])
                .repeat_dim(0, t);
            let targets = lab_t.clone().select(0, idx_t);
            let logits = es_forward_factored(
                images,
                &weights_inner,
                &factors,
                &cfg,
                true, true, beta_t,
                Some(&qam_run),
            );
            let logsm = log_softmax2(logits);
            let raw = logsm
                .gather(1, targets.reshape([cand_slice.len(), 1]))
                .reshape([cand_slice.len()]);
            // 梯度累积（仅 ES 槽有非零噪声；偏置/lora 槽累加的是零）
            for di in 0..dense_slots.len() {
                let noise = &factors.dense[di];
                let g = noise.clone().transpose().matmul(raw.clone().reshape([cand_slice.len(), 1]));
                grad_acc[lora_slots.len() + di] = grad_acc[lora_slots.len() + di].clone() + g;
            }
            raw_sum_t = raw_sum_t.clone() + raw.clone().sum();
            raw_sumsq_t = raw_sumsq_t.clone() + raw.powf_scalar(2.0).sum();
            n_used += cand_slice.len();
        }

        // z-score + AdamW（wd=0）
        let s1v = raw_sum_t.into_scalar();
        let s2v = raw_sumsq_t.into_scalar();
        let mean = s1v / n_used as f32;
        let var = (s2v / n_used as f32 - mean * mean).max(0.0);
        let stdv = (var + 1e-5).sqrt();
        let scale = -1.0 / (stdv * (n_used as f32).sqrt());
        let grads: Vec<Tensor<B, 2>> =
            grad_acc.into_iter().map(|g| g.mul_scalar(scale)).collect();
        optim_es.step(&mut es_params, &grads);
        es_loglik = mean as f64;

        // v_th 回读（8 次标量同步/epoch）
        for (k, site) in vth_sites.iter().enumerate() {
            let idx = es_slot_base + qam_sites.len() * 2 + k;
            let v = es_params[idx].clone().into_scalar() as f64;
            if let Some(e) = vth_cpu.iter_mut().find(|(s, _)| s == site) {
                e.1 = v;
            }
        }
        <B as burn::tensor::backend::Backend>::sync(device).expect("GPU 同步失败");
        <B as burn::tensor::backend::Backend>::memory_cleanup(device);

        // ---- 3) 验证（硬前向 + 当前 v_th/QAM）----
        let mut val_str = String::new();
        if epoch % args.validate_every == 0 || epoch == args.epochs - 1 {
            let weights_inner2 = to_wgpu_weights(&model.to_weights());
            let qam_eval = if qam_sites.is_empty() && vth_cpu.is_empty() {
                None
            } else {
                let mut sites = Vec::new();
                for (k, &site) in qam_sites.iter().enumerate() {
                    let ia = es_slot_base + 2 * k;
                    let na = es_params[ia].dims()[0];
                    sites.push((
                        site,
                        es_params[ia].clone().reshape([na]),
                        es_params[ia + 1].clone().reshape([na]),
                    ));
                }
                Some(QamWeights { sites, vth: vth_cpu.clone() })
            };
            let val_top1 = eval_es(&weights_inner2, qam_eval.as_ref(), &data, 256, t, &cfg);
            if val_top1 > best_val {
                best_val = val_top1;
            }
            val_str = format!("{val_top1:.2}");
            println!(
                "epoch={}/{}, sgd_loss={:.4}, es_loglik={:.4}, val_top1={:.2}%, best_val={:.2}%（本轮 {:.1}s，lr_sgd={:.5}，v_th={:?}）",
                epoch + 1, args.epochs, sgd_loss, es_loglik, val_top1, best_val,
                ep_start.elapsed().as_secs_f32(), lr_sgd_t,
                vth_cpu.iter().map(|(_, v)| format!("{v:.2}")).collect::<Vec<_>>()
            );
        } else {
            println!(
                "epoch={}/{}, sgd_loss={:.4}, es_loglik={:.4}（本轮 {:.1}s，lr_sgd={:.5}）",
                epoch + 1, args.epochs, sgd_loss, es_loglik,
                ep_start.elapsed().as_secs_f32(), lr_sgd_t
            );
        }

        let el = ep_start.elapsed().as_secs_f64();
        cum_t += el;
        csv_rows.push(format!(
            "{},{:.6},{:.6},{},{:.2},{:.1},{:.1}",
            epoch + 1, sgd_loss, es_loglik, val_str, best_val, el, cum_t
        ));
        write_csv(&args.csv_out, &csv_rows);
    }

    println!("混合估计器训练完成：best_val={best_val:.2}% 总耗时 {cum_t:.0}s");
    println!("训练指标已写入: {}", args.csv_out);
}
