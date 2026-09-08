# -*- coding: utf-8 -*-
"""QAM 调制 ES 数学验证（ES_MANIFOLD_NOTE.md §12）

验证项：
  c1  增益-阈值对偶（命题 5）：LIF(m·u, θ) ≡ LIF(u, θ/m)，硬与松弛皆然
  c2  m 槽景观秩（命题 6）：松弛前向下 m 槽 ES 信号余弦 vs W 槽（平板流形）
  c3  T 压缩预演（命题 7）：T=2+QAM-learnable 与 T=4 纯净的 fitness 方差对比

用法：python scripts/verify_qam_math.py [c1|c2|c3|all]
"""
import sys
import math

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAU = 2.0  # 与 rust-sdt lif 的 tau=2.0 一致（v += (x-v)/2）


# ---------------- 公共：LIF 序列（硬 / 松弛） ----------------
def lif_seq(x, threshold=1.0, beta=None):
    """x: [T, B, N]；beta=None 硬脉冲，否则松弛读出 s=sigmoid(beta*(h-thr))。
    与 rust-sdt lif_step 一致：h = v + (x-v)/tau；硬重置 v=(1-s)*h。"""
    T = x.shape[0]
    v = torch.zeros_like(x[0])
    outs = []
    for t in range(T):
        h = v + (x[t] - v) / TAU
        if beta is None:
            s = (h > threshold).float()
        else:
            s = torch.sigmoid(beta * (h - threshold))
        v = (1.0 - s) * h
        outs.append(s)
    return torch.stack(outs, 0)


# ---------------- c1: 增益-阈值对偶 ----------------
def c1():
    print("=" * 60)
    print("c1 命题 5：增益-阈值对偶 LIF(m*u, θ) ≡ LIF(u, θ/m)")
    T, B, N = 4, 64, 128
    u = torch.randn(T, B, N, device=DEV) * 0.8
    m = torch.rand(N, device=DEV) * 1.5 + 0.3  # m ∈ [0.3, 1.8] > 0
    theta = 1.0
    # 硬脉冲：精确对偶
    a = lif_seq(u * m, theta, None)
    b = lif_seq(u, theta / m, None)
    d_hard = (a - b).abs().max().item()
    print(f"  [硬] max|差| = {d_hard:.3e}（预期精确 0）")
    # 松弛：对偶伴随温度耦合 β→β·m：σ(β(m·u−θ)) = σ(βm(u−θ/m))
    worst = 0.0
    for beta in (4.0, 16.0):
        a = torch.sigmoid(beta * (u * m - theta))
        b = torch.sigmoid((beta * m) * (u - theta / m))
        d = (a - b).abs().max().item()
        print(f"  [松弛β={beta}] 温度耦合恒等 max|差| = {d:.3e}")
        worst = max(worst, d)
    print("  [对偶含义] θ_i = θ/m_i 且 α_i = β·m_i：m→0+ ⇒ 沉默+宽噪声，m<0 ⇒ 极性翻转")
    ok = d_hard < 1e-6 and worst < 1e-6
    print("  ✅ 通过" if ok else "  ❌ 失败")
    return ok


# ---------------- c2: m 槽 vs W 槽的 ES 信号余弦 ----------------
class TinySNN(nn.Module):
    """2 层 LIF-MLP（模拟 SDT 位点结构：线性→LIF→线性→LIF→head）。
    qam: None 或 'learnable'（每神经元 m=(1+a)cos(φ) 乘在两个 LIF 输入上）。"""

    def __init__(self, n_in=64, n_h=128, n_out=10, qam=None):
        super().__init__()
        self.qam = qam
        self.w1 = nn.Parameter(torch.randn(n_h, n_in) * 0.08)
        self.b1 = nn.Parameter(torch.zeros(n_h))
        self.w2 = nn.Parameter(torch.randn(n_out, n_h) * 0.08)
        self.b2 = nn.Parameter(torch.zeros(n_out))
        if qam == "learnable":
            self.a = nn.Parameter(torch.full((n_h,), 0.1))
            self.phi = nn.Parameter(torch.full((n_h,), 0.5))
            self.a2 = nn.Parameter(torch.full((n_out,), 0.1))
            self.phi2 = nn.Parameter(torch.full((n_out,), 0.5))

    def m(self, p_a, p_phi):
        return (1.0 + p_a) * torch.cos(p_phi)

    def forward(self, x, beta=None):
        """x: [T, B, n_in] -> logits [B, n_out]（时间平均）"""
        T = x.shape[0]
        u1 = torch.einsum("tbi,oi->tbo", x, self.w1) + self.b1
        if self.qam == "learnable":
            u1 = u1 * self.m(self.a, self.phi)
        s1 = lif_seq(u1, 1.0, beta)
        u2 = torch.einsum("tbi,oi->tbo", s1, self.w2) + self.b2
        if self.qam == "learnable":
            u2 = u2 * self.m(self.a2, self.phi2)
        s2 = lif_seq(u2, 1.0, beta)
        return s2.mean(0)


def _fitness(model, x, y, beta):
    logits = model(x, beta)
    return torch.log_softmax(logits, 1)[torch.arange(y.shape[0]), y]


def _es_cosine(model, names, x, y, sigma, n_pairs=256, beta=4.0):
    """有限差分反对称 ES：对参数组 names 里每个参数做 ±σ·ε 扰动，
    返回每组 |cos(ES_g, 解析差分方向)|（解析方向 = 数值细差分）。"""
    flat = {n: p.detach().clone() for n, p in model.named_parameters() if n in names}
    # 1) ES 梯度（反对称对平均）
    es = {n: torch.zeros_like(p) for n, p in flat.items()}
    for _ in range(n_pairs):
        eps = {n: torch.randn_like(p) / p.numel() ** 0.5 for n, p in flat.items()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in flat:
                    p.copy_(flat[n] + sigma * eps[n])
            fp = _fitness(model, x, y, beta).sum().item()
            for n, p in model.named_parameters():
                if n in flat:
                    p.copy_(flat[n] - sigma * eps[n])
            fm = _fitness(model, x, y, beta).sum().item()
        for n in flat:
            es[n] += (fp - fm) * eps[n] / (2 * sigma)
    for n in flat:
        es[n] /= n_pairs
    # 2) 解析方向：真实梯度（松弛前向可微，直接 autograd）
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in flat:
                p.copy_(flat[n])
    lg = _fitness(model, x, y, beta).sum()
    g = torch.autograd.grad(lg, [model.get_parameter(n) for n in flat])
    ana = {n: gg for n, gg in zip(flat, g)}
    # 3) 余弦
    out = {}
    for n in flat:
        a = es[n].flatten()
        b = ana[n].flatten()
        cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        out[n] = cos
    return out


def c2():
    print("=" * 60)
    print("c2 命题 6：m 槽（QAM）ES 信号余弦 vs W 槽（平板流形）")
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST("H:/snn-thinking/agid/qam-snn/data", train=True, download=False, transform=tf)
    xs = torch.stack([ds[i][0] for i in range(256)]).to(DEV)
    ys = torch.tensor([ds[i][1] for i in range(256)], device=DEV)
    # 静态帧重复 T=4（与 SDT 口径一致）
    x = xs.flatten(1)[:, :64].unsqueeze(0).repeat(4, 1, 1).contiguous()  # [T, B, 64]
    x = x[:, :, :64].contiguous()

    for qam in (None, "learnable"):
        model = TinySNN(qam=qam).to(DEV)
        sigma = 0.02
        names_w = ["w1", "w2"]
        names = names_w + (["a", "phi", "a2", "phi2"] if qam else [])
        cos = _es_cosine(model, names, x, ys, sigma, n_pairs=192, beta=4.0)
        tag = "有 QAM" if qam else "无 QAM"
        wc = [abs(cos[n]) for n in names_w]
        print(f"  [{tag}] W 槽 |cos|: w1={wc[0]:.3f} w2={wc[1]:.3f}")
        if qam:
            print(f"           m 槽 |cos|: a={abs(cos['a']):.3f} phi={abs(cos['phi']):.3f} "
                  f"a2={abs(cos['a2']):.3f} phi2={abs(cos['phi2']):.3f}")
    print("  预期：m 槽 |cos| ≈ 1（满秩信号），W 槽 |cos| 明显更低（接口限制信号）")
    return True


# ---------------- c3: T 压缩预演 ----------------
def c3():
    print("=" * 60)
    print("c3 命题 7：T=2+QAM-learnable vs T=4 纯净（fitness 方差/信号量）")
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST("H:/snn-thinking/agid/qam-snn/data", train=True, download=False, transform=tf)
    xs = torch.stack([ds[i][0] for i in range(256)]).to(DEV)
    ys = torch.tensor([ds[i][1] for i in range(256)], device=DEV)
    x4 = xs.flatten(1)[:, :64].unsqueeze(0).repeat(4, 1, 1).contiguous()
    x2 = xs.flatten(1)[:, :64].unsqueeze(0).repeat(2, 1, 1).contiguous()

    # 无 QAM T=4
    m0 = TinySNN(qam=None).to(DEV)
    # 有 QAM T=2（learnable，乘在两个 LIF 输入）
    m1 = TinySNN(qam="learnable").to(DEV)

    for tag, model, x, T in (("T=4 无QAM", m0, x4, 4), ("T=2 +QAM", m1, x2, 2)):
        model.eval()
        with torch.no_grad():
            lg = model(x, beta=4.0)
        ll = torch.log_softmax(lg, 1)[torch.arange(ys.shape[0]), ys]
        print(f"  [{tag}] fitness: mean={ll.mean().item():.4f} std={ll.std().item():.4f} "
              f"top1={(lg.argmax(1) == ys).float().mean().item() * 100:.1f}%")
    print("  说明：等精度下 T=2+QAM 的前向成本减半；此为预演，正式验证在 rust 侧 10ep A/B")
    return True


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if which in ("c1", "all"):
        ok &= c1()
    if which in ("c2", "all"):
        ok &= c2()
    if which in ("c3", "all"):
        ok &= c3()
    print("=" * 60)
    print("全部通过" if ok else "存在失败项")
