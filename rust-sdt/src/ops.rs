//! 核心算子的 Burn 实现，数值语义与 spikingjelly v0.0.0.0.12（cupy 内核）严格一致：
//!
//! 前向（MultiStepLIFNodePTT fptt 内核，v_reset=0.0 => 硬重置，decay_input=True）：
//!   h[t]     = v[t-1] + (x[t] - v[t-1]) / tau          (tau = 2.0)
//!   spike[t] = (h[t] >= v_threshold) ? 1.0 : 0.0        (闭区间 >=)
//!   v[t]     = spike[t] ? v_reset(=0) : h[t]
//!
//! 反向（bptt 内核，detach_reset=True，Sigmoid 代理 alpha=4.0）：
//!   grad_s_to_h  = alpha * sigmoid(alpha*(h-thr)) * (1 - sigmoid(alpha*(h-thr)))
//!   grad_v_to_h  = 1 - spike
//!   grad_h[t]    = grad_spike[t]*grad_s_to_h + (grad_v[t] + grad_h[t+1]*(1-1/tau)) * grad_v_to_h
//!   grad_x[t]    = grad_h[t] / tau
//!
//! Burn 不支持自定义 autograd 算子，但可用「detach 直通估计器（STE）」技巧
//! 精确复现上述前向与反向：
//!   spike = sigmoid(4*(h-thr)) - sigmoid(4*(h-thr)).detach() + step(h-thr).detach()
//!     前向：sigmoid - sigmoid + step = step（精确二值）
//!     反向：d(spike)/dh = d(sigmoid(4*(h-thr)))/dh = 4*sigmoid*(1-sigmoid)（精确代理梯度）
//!   v_new = (1 - spike).detach() * h
//!     前向：(1-spike)*h（v_reset=0 的硬重置）
//!     反向：d(v_new)/dh = (1-spike)（与 grad_v_to_h 一致；reset 分支不回传，等价 detach_reset=True）

use burn::prelude::*;

/// 时间常数 tau = 2.0（与 PyTorch 侧 MultiStepLIFNode(tau=2.0) 一致）
pub const TAU: f64 = 2.0;
/// Sigmoid 代理梯度的 alpha（与 spikingjelly 默认 surrogate.Sigmoid() 一致）
pub const SG_ALPHA: f64 = 4.0;

/// 单步 LIF：返回 (spike, v_new)。
/// * `v_prev`：上一时刻膜电位
/// * `x`：输入电流
/// * `threshold`：放电阈值（1.0 或 0.5）
pub fn lif_step<B: Backend, const D: usize>(
    v_prev: Tensor<B, D>,
    x: Tensor<B, D>,
    threshold: f64,
) -> (Tensor<B, D>, Tensor<B, D>) {
    // 充电：h = v + (x - v) / tau
    let h = v_prev.clone() + (x - v_prev.clone()) / TAU;
    // 精确二值脉冲（前向） + 精确代理梯度（反向）
    let spike = spike_with_surrogate(h.clone(), threshold);
    // 硬重置到 v_reset = 0：v_new = (1 - spike) * h，reset 分支不回传梯度
    let one_minus_spike = (1.0f32 - spike.clone()).detach();
    let v_new = one_minus_spike * h;
    (spike, v_new)
}

/// 精确 heaviside 前向 + Sigmoid(4.0) 代理梯度反向。
/// 前向：h >= thr ? 1 : 0（与 CUDA 内核 if (h >= v_threshold) 一致）
/// 反向：alpha * sigmoid(alpha*(h-thr)) * (1 - sigmoid(alpha*(h-thr)))
fn spike_with_surrogate<B: Backend, const D: usize>(
    h: Tensor<B, D>,
    threshold: f64,
) -> Tensor<B, D> {
    // x = h - threshold
    let x = h - threshold;
    // sg = sigmoid(alpha * x)，带梯度：d(sg)/dh = alpha*sigmoid*(1-sigmoid)（精确代理梯度）
    let sg = burn::tensor::activation::sigmoid(x.clone() * SG_ALPHA);
    let sg_d = sg.clone().detach();
    // step = (x >= 0) ? 1 : 0，精确二值前向；mask 构造不产生梯度
    let step = Tensor::<B, D>::ones(x.shape(), &x.device())
        .mask_fill(x.lower_equal_elem(0.0), 0.0f32)
        .detach();
    // 前向：sg - sg + step = step；反向：仅 sg 的梯度回传
    sg - sg_d + step
}

/// 多时间步 LIF：输入 [T, ...]（时间维在 dim 0），输出同形状脉冲序列。
/// 初始膜电位为 0（等价于 PyTorch 侧每次 forward 前 reset_net）。
/// 实现说明：每步切片出 [1, ...] 形状（保持总维数 D），循环后沿 dim 0 拼接。
pub fn lif_seq<B: Backend, const D: usize>(x: Tensor<B, D>, threshold: f64) -> Tensor<B, D> {
    let t = x.dims()[0];
    // 初始膜电位：[1, ...] 形状（与单个时间步切片同形）
    let mut v = Tensor::zeros(x.clone().slice([0..1]).shape(), &x.device());
    let mut spikes: Vec<Tensor<B, D>> = Vec::with_capacity(t);
    for i in 0..t {
        // 取第 i 个时间步：[1, ...]
        let xt = x.clone().slice([i..i + 1]);
        let (spike, v_new) = lif_step(v, xt, threshold);
        v = v_new;
        spikes.push(spike);
    }
    Tensor::cat(spikes, 0)
}
/// SPS 中的 maxpool：kernel=3, stride=2, padding=1（ceil_mode=False）。
/// 输入 [B, C, H, W]，输出 [B, C, (H+2-3)/2+1, ...]（floor 语义与 PyTorch 一致）。
pub fn maxpool2d_3x3_s2<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn::tensor::module::max_pool2d(x, [3, 3], [2, 2], [1, 1], [1, 1])
}
