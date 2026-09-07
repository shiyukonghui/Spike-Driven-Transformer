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
use crate::loader::{Cifar10Npz, Split};
use crate::model::{
    forward_full, forward_sps, reshape_heads, unshape_heads, BackendAdapter, BlockWeights,
    ConvLayer, Dev, SdtWeights,
};
use crate::ops::{lif_seq, lif_seq_relaxed};

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
}

/// 内层 wgpu 设备引用
fn es_dev() -> &'static Dev {
    static ONCE: std::sync::OnceLock<Dev> = std::sync::OnceLock::new();
    ONCE.get_or_init(Default::default)
}

/// 验证集 top-1（内层后端分批推理，逐批 sync + cleanup）
fn eval_es(w: &SdtWeights<B>, data: &Cifar10Npz, batch: usize, t: usize, cfg: &SdtConfig) -> f64 {
    let device = es_dev();
    let n = data.n_test;
    let order: Vec<usize> = (0..n).collect();
    let mut correct = 0usize;
    for chunk in order.chunks(batch) {
        let (images, targets) = data.get_batch(Split::Test, chunk, t, device);
        let logits = forward_full(images, w, cfg);
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
                let logits = forward_full(images, &pert, &cfg); // [m, C]

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
            let val_top1 = eval_es(&w_now, &data, 64, args.time_steps, &cfg);
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FLoc {
    BlkW(usize, usize),
    BlkB(usize, usize),
    HeadW,
    HeadB,
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
fn factored_slots(base: &SdtWeights<B>, depths: usize) -> (Vec<FSlotSpec>, Vec<FSlotSpec>) {
    let mut lora = Vec::new();
    let mut dense = Vec::new();
    let mut ki = 0u64;
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
) -> Tensor<B, 2> {
    let t = images.dims()[0];
    let c = images.dims()[1];
    let device = es_dev();
    // S6：松弛作用域——ssa 模式只松弛 SSA 位点（xs/q/k/v/kv，梯度信号集中处），
    // MLP 位点（x1/x2）与 head 用硬 LIF；all 模式全部松弛（与 v4 行为一致）
    let lif = |x: Tensor<B, 5>, th: f64| -> Tensor<B, 5> {
        if relax_ssa {
            lif_seq_relaxed(x, th, beta as f64)
        } else {
            lif_seq(x, th)
        }
    };
    let lif_mlp = |x: Tensor<B, 5>, th: f64| -> Tensor<B, 5> {
        if relax_mlp {
            lif_seq_relaxed(x, th, beta as f64)
        } else {
            lif_seq(x, th)
        }
    };
    let lif4 = |x: Tensor<B, 4>, th: f64| -> Tensor<B, 4> {
        if relax_mlp {
            lif_seq_relaxed(x, th, beta as f64)
        } else {
            lif_seq(x, th)
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
        let xs = lif(x.clone(), 1.0);
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

        let xp = noisy_conv1x1(xo, &w2(3), bias5(3, ch), &lw(3).0, &lw(3).1);
        let ssa_out = xp + identity;

        // MLP
        let identity2 = ssa_out.clone();
        let x1 = lif_mlp(ssa_out, 1.0);
        let hidden = cfg.mlp_hidden();
        let x1c = noisy_conv1x1(x1, &w2(4), bias5(4, hidden), &lw(4).0, &lw(4).1);
        let x2 = lif_mlp(x1c, 1.0);
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
    let feat_spike = lif4(feat, 1.0); // [T, C, 256, 1]（sum_dim 保留维）
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

/// factored 主参数列表（lora 13 槽在前、dense 13 槽在后，顺序与 factored_slots 一致）
fn factored_params2d(base: &SdtWeights<B>, lora: &[FSlotSpec], dense: &[FSlotSpec]) -> Vec<Tensor<B, 2>> {
    let mut out = Vec::new();
    for slot in lora {
        let t = match slot.loc {
            FLoc::BlkW(j, which) => flatten_slot(&slot.shape, floc_get4(base, j, which)),
            FLoc::HeadW => flatten_slot(&slot.shape, &base.head_w),
            _ => unreachable!(),
        };
        out.push(t);
    }
    for slot in dense {
        let t = match slot.loc {
            FLoc::BlkB(j, which) => flatten_slot(&slot.shape, floc_get1(base, j, which)),
            FLoc::HeadB => flatten_slot(&slot.shape, &base.head_b),
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
            _ => unreachable!(),
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
            _ => unreachable!(),
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
    let (lora_slots, dense_slots) = factored_slots(&base, cfg.depths);
    let mut params2d = factored_params2d(&base, &lora_slots, &dense_slots);
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
        let mut sigma_lora = Vec::with_capacity(lora_slots.len());
        let mut sigma_dense = Vec::with_capacity(dense_slots.len());
        for (i, p) in params2d.iter().enumerate() {
            let n = p.dims()[0] * p.dims()[1];
            let mean = p.clone().sum().into_scalar() / n as f32;
            let var = (p.clone().powf_scalar(2.0).sum().into_scalar() / n as f32 - mean * mean).max(0.0);
            let s = args.sigma * var.sqrt();
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
        for k in 0..n_chunks {
            // ---- 1) S3：GPU 端因子生成（零 CPU 生成、零 PCIe 上传）----
            let t0 = std::time::Instant::now();
            let cand_slice: Vec<usize> = order_vec[k * args.chunk..(k + 1) * args.chunk].to_vec();
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

            // ---- 2) 因式噪声前向 + fitness ----
            let t1 = std::time::Instant::now();
            // S2：GPU 行选择替代 get_batch（语义与 [T,B,3,32,32] 逐位一致）
            let idx_t = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(
                    cand_slice.iter().map(|&i| i as i64).collect::<Vec<i64>>(),
                    [cand_slice.len()],
                ),
                device,
            );
            let onehot = idx_t.clone().reshape([cand_slice.len(), 1]).equal(ar_t.clone()).float(); // [C,N]
            let sel = onehot.matmul(pix_t.clone()); // [C, 3072]（f32 精确行拷贝）
            let images = sel
                .reshape([cand_slice.len(), 3, 32, 32])
                .reshape([1, cand_slice.len(), 3, 32, 32])
                .repeat_dim(0, args.time_steps); // [T, C, 3, 32, 32]
            let targets = lab_t.clone().select(0, idx_t); // [C] Int
            // S4 优化：β 退火到 16 后切硬 LIF——Run-1（1000ep）实测为训练质量负优化
            // （峰值后衰减回归，机制 = 定理 2/3：硬 fitness 信号只在切换复形上），
            // 故改为 opt-in：默认全程松弛（v4 行为），--hard-at-16 启用硬切换。
            // S6：relax-scope=ssa 时 MLP/head 位点保持硬 LIF
            let relax_fwd = args.relax && (beta_t < 16.0 || !args.hard_at_16);
            let relax_mlp = relax_fwd && args.relax_scope != "ssa";
            let logits = es_forward_factored(images, &base, &factors, &cfg, relax_fwd, relax_mlp, beta_t); // [C, 10]
            let logsm = log_softmax2(logits.clone());
            // S7 优化：one_hot 构造+乘法 → gather（raw[c] = logsm[c, targets[c]]，逐位一致）
            let raw = logsm
                .gather(1, targets.clone().reshape([cand_slice.len(), 1]))
                .reshape([cand_slice.len()]); // [C]
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
        optim.step(&mut params2d, &grads);
        write_back_factored(&mut base, &lora_slots, &dense_slots, &params2d);

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
            let val_top1 = eval_es(&base, &data, 256, args.time_steps, &cfg); // S9：批 64→256
            if val_top1 > best_val {
                best_val = val_top1;
            }
            val_str = format!("{val_top1:.2}");
            println!(
                "epoch={}/{}, train_top1={:.2}%, train_loglik={:.4}, val_top1={:.2}%, best_val={:.2}%{}（gen {:.1}s fwd {:.1}s，本轮 {:.1}s，lr={:.5}）",
                epoch + 1, args.epochs, train_top1, train_loglik, val_top1, best_val,
                if args.relax {
                    if args.relax && beta_t < 16.0 {
                        format!(" β={beta_t:.1}(松弛前向)")
                    } else {
                        format!(" β={beta_t:.1}(硬前向)")
                    }
                } else {
                    String::new()
                },
                gen_time, fwd_time,
                ep_start.elapsed().as_secs_f32(),
                lr_t
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
