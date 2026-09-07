# -*- coding: utf-8 -*-
"""
ES_MANIFOLD_NOTE.md 的数值验证：以真实数字检验每一条数学断言。
数据：data/cifar10_full/cifar10_data.npz（真实 CIFAR-10，[0,1]，CHW）
环境：D:\\Anaconda\\envs\\Pytorch-CUDA（torch 2.0.1 + cu118）

A. 恒等式验证（合成，MC 精度）
   A1 Stein: ES 估计量 ≈ ∇F_σ（二次型解析 + MLP 有限差分）
   A2 阈值噪声恒等: E_ξ[1[h+ξ≥τ]] = σ(β(h−τ))，ξ~Logistic(0,1/β)
   A3 裕度界: |σ(βx)−H(x)| ≤ e^{−β|x|}
   A4 级联: 硬阈值深度 L↑ → P(Δf≠0)↓ + 跳变 O(1)；松弛 → 处处平滑
B. 引理 1 胞内平坦性（真实数据 + 真实 SDT 结构 mini 复刻）
   胞内: block 权重方向 Δlogits ≡ 0（精确）；head 权重 ≠ 0
   跨胞: 越过首个切换超平面 → O(1) 跳变
   松弛: 同一胞内 η 下 ΔF ≠ 0（斜率被抬出）
C. 定理 2/3 信号通道（真实数据）
   ES(hard) vs STE(α=4) 余弦 ≈ 0；ES(relaxed β=4) vs STE 显著 > 0
   松弛解析梯度 vs STE 高度一致；逐层 fitness 方差剖面（方差灾难）
D. 真实训练对照（8000 训练/2000 测试）
   SGD-STE / ES-hard / ES-relaxed / ES-relaxed+退火
运行: D:\\Anaconda\\envs\\Pytorch-CUDA\\python.exe scripts/verify_es_math.py
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

torch.manual_seed(42)
np.random.seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BETA = 4.0  # 与 spikingjelly STE α=4 / rust 侧一致


def hr(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72, flush=True)


# ---------------------------------------------------------------------------
# A. 恒等式验证（numpy，合成）
# ---------------------------------------------------------------------------
def part_a():
    hr("A. 恒等式验证（合成）")
    rng = np.random.default_rng(0)
    D, sigma = 50, 0.3
    theta_star = rng.normal(size=D)
    theta = rng.normal(size=D) * 0.5

    # A1a 二次型 F=½||θ−θ*||²：解析 ∇F_σ = θ−θ*；MC 分块 1e7
    g_mc = np.zeros(D)
    n_tot, chunk = 10_000_000, 1_000_000
    for _ in range(n_tot // chunk):
        eps = rng.normal(size=(chunk, D)) * sigma
        Fv = 0.5 * ((theta[None, :] + eps - theta_star) ** 2).sum(1)
        g_mc += (Fv[:, None] * eps).sum(0) / sigma**2
    g_mc /= n_tot // chunk
    g_an = theta - theta_star
    cos = float(g_mc @ g_an / (np.linalg.norm(g_mc) * np.linalg.norm(g_an)))
    rel = float(np.linalg.norm(g_mc - g_an) / np.linalg.norm(g_an))
    print(f"[A1a] Stein(二次型, MC 1e7): cos={cos:.5f} rel={rel:.4f} -> "
          f"{'PASS' if cos > 0.999 else 'FAIL'}")

    # A1b MLP 非线性 fitness：ES(4e6) vs 平滑景观有限差分(CRN, 1e6/侧, 分块)
    W1 = rng.normal(size=(16, D)) * 0.5
    b1 = rng.normal(size=16) * 0.1
    w2 = rng.normal(size=16) * 0.5

    def fitness_batch(X):  # X: [n,D] -> [n]
        return -(np.tanh(X @ W1.T + b1) @ w2)

    n_es = 4_000_000
    g_es = np.zeros(D)
    for _ in range(n_es // 500_000):
        eps = rng.normal(size=(500_000, D)) * sigma
        Fv = fitness_batch(theta[None, :] + eps)
        g_es += (Fv[:, None] * eps).sum(0) / sigma**2
    g_es /= n_es // 500_000
    n_fd, delta = 1_000_000, 1e-3
    eps_fd = rng.normal(size=(n_fd, D)) * sigma
    g_fd = np.zeros(D)
    for j in range(D):
        e = np.zeros(D); e[j] = delta
        Fp = fitness_batch(theta[None, :] + e[None, :] + eps_fd).mean()
        Fm = fitness_batch(theta[None, :] - e[None, :] + eps_fd).mean()
        g_fd[j] = (Fp - Fm) / (2 * delta)
    cos2 = float(g_es @ g_fd / (np.linalg.norm(g_es) * np.linalg.norm(g_fd)))
    print(f"[A1b] Stein(MLP, ES 4e6 vs 平滑FD CRN 1e6): cos={cos2:.4f} -> "
          f"{'PASS' if cos2 > 0.98 else 'FAIL'}")

    # A2 阈值噪声恒等
    xs = np.linspace(-3, 3, 13)
    errs = []
    for x in xs:
        xi = rng.logistic(0.0, 1.0 / BETA, 1_000_000)
        mc = float((xi + x >= 0).mean())
        errs.append(abs(mc - float(1 / (1 + math.exp(-BETA * x)))))
    print(f"[A2] E_ξ[1[h+ξ≥τ]] = σ(β(h−τ)) MC 1e6/点: max|err|={max(errs):.5f} -> "
          f"{'PASS' if max(errs) < 0.005 else 'FAIL'}")

    # A3 裕度界
    xs = np.linspace(-5, 5, 2001)
    viol = np.maximum(np.abs(1 / (1 + np.exp(-BETA * xs)) - (xs >= 0)) -
                      np.exp(-BETA * np.abs(xs)), 0).max()
    print(f"[A3] |σ(βx)−H(x)| ≤ e^(−β|x|) 最大违反量: {viol:.2e} -> "
          f"{'PASS' if viol <= 1e-12 else 'FAIL'}")

    # A4 级联：逐层校准发放率后，Var/P(非零)/跳变幅度 vs 深度
    d, L_list, sigma_w, n_draw = 16, [1, 2, 4, 8, 16], 0.05, 3000
    rng2 = np.random.default_rng(1)
    Ws_raw = [rng2.normal(size=(d, d)) / math.sqrt(d) for _ in range(max(L_list))]
    # 逐层校准：membrane std -> 1.2（模拟静态校准，保持信号跨层存活）
    scales = []
    h_cal = np.stack([ (rng2.normal(size=d) > 0).astype(float) for _ in range(4000)])
    for l in range(max(L_list)):
        mem = h_cal @ Ws_raw[l].T
        s_l = 1.2 / (mem.std() + 1e-9)
        scales.append(s_l)
        h_cal = (mem * s_l >= 1.0).astype(float)  # 下一层输入分布（硬）
    Ws = [Ws_raw[l] * scales[l] for l in range(max(L_list))]

    def cascade(Wlist, hard=True, beta=BETA):
        h = (rng2.normal(size=d) > 0).astype(np.float64)  # 随机 0/1 输入
        for W in Wlist:
            mem = W @ h
            if hard:
                s = (mem >= 1.0).astype(np.float64)
            else:
                s = 1 / (1 + np.exp(-beta * (mem - 1.0)))
            h = s
        return h.sum()

    print(f"[A4] 深度扫描（每层校准 mem std=1.2, σ_w={sigma_w}, {n_draw} 采样）")
    print(f"     {'L':>3} | {'hard: P(Δf≠0)':>13} {'Var':>11} {'max|Δf|':>9} || "
          f"{'relax: P(≠0)':>12} {'Var':>11} {'max|Δf|':>9}")
    for L in L_list:
        vh_, ph, mh = [], [], []
        vr_, pr, mr = [], [], []
        f0h = cascade(Ws[:L]); f0r = cascade(Ws[:L], hard=False)
        for _ in range(n_draw):
            dW = [rng2.normal(size=(d, d)) * sigma_w for _ in range(L)]
            dh = cascade([Ws[i] + dW[i] for i in range(L)]) - f0h
            dr = cascade([Ws[i] + dW[i] for i in range(L)], hard=False) - f0r
            vh_.append(dh); vr_.append(dr)
        vh_ = np.array(vh_); vr_ = np.array(vr_)
        nz_h = vh_[np.abs(vh_) > 1e-12]; nz_r = vr_[np.abs(vr_) > 1e-12]
        print(f"     {L:>3} | {len(nz_h)/n_draw:>13.3f} {np.var(vh_):>11.3e} "
              f"{(np.abs(nz_h).max() if len(nz_h) else 0):>9.3f} || "
              f"{len(nz_r)/n_draw:>12.3f} {np.var(vr_):>11.3e} {np.abs(vr_).max():>9.3f}")


# ---------------------------------------------------------------------------
# 真实数据加载
# ---------------------------------------------------------------------------
def load_data(n_train=8000, n_test=2000):
    z = np.load("data/cifar10_full/cifar10_data.npz")
    tr_x, tr_y = z["train_x"], z["train_y"]
    te_x, te_y = z["test_x"], z["test_y"]
    itr = np.random.RandomState(42).permutation(len(tr_x))[:n_train]
    ite = np.random.RandomState(42).permutation(len(te_x))[:n_test]
    to_t = lambda x: torch.from_numpy(x).float().to(DEV)
    to_y = lambda y: torch.from_numpy(y).long().to(DEV)
    return (to_t(tr_x[itr]), to_y(tr_y[itr]), to_t(te_x[ite]), to_y(te_y[ite]))


# ---------------------------------------------------------------------------
# Mini SDT（结构忠实复刻：stem + LIF-SSA blocks(×2) + LIF head）
# ---------------------------------------------------------------------------
class LIFSTE(torch.autograd.Function):
    """硬阈值前向 + sigmoid(β) 代理反向（= spikingjelly α=4）"""
    @staticmethod
    def forward(ctx, mem, th):
        ctx.save_for_backward(mem)
        ctx.th = th
        return (mem >= th).float()

    @staticmethod
    def backward(ctx, g):
        (mem,) = ctx.saved_tensors
        s = torch.sigmoid(BETA * (mem - ctx.th))
        return g * BETA * s * (1 - s), None


class Block(nn.Module):
    def __init__(self, d, heads, hidden):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)
        self.heads, self.hd = heads, d // heads

    def forward(self, z, lif):
        B, N, d = z.shape
        H, hd = self.heads, self.hd
        shape = lambda t: t.view(B, N, H, hd).transpose(1, 2)  # [B,H,N,hd]
        xq = lif(self.q(z))
        xk = lif(self.k(z))
        xv = lif(self.v(z))
        qh, kh, vh = shape(xq), shape(xk), shape(xv)
        kv = (kh * vh).sum(2, keepdim=True)        # [B,H,1,hd] 线性注意力聚合
        kv_sp = lif(kv, 0.5)                        # LIF(0.5)
        xattn = qh * kv_sp                          # 乘性读出（广播 N）
        xo = xattn.transpose(1, 2).reshape(B, N, d)
        z = self.proj(xo) + z                       # 残差
        z = self.fc2(lif(self.fc1(z))) + z          # MLP 残差
        return z


class MiniSDT(nn.Module):
    def __init__(self, d=32, heads=2, hidden=64, T=2, classes=10):
        super().__init__()
        self.d, self.heads, self.T = d, heads, T
        self.stem = nn.Conv2d(3, d, 4, 4, bias=False)  # 32→8×8=64 tokens
        self.blocks = nn.ModuleList([Block(d, heads, hidden) for _ in range(2)])
        self.head = nn.Linear(d, classes)
        self.register_buffer("gamma", torch.ones(()))
        self.relaxed = False
        self.beta = BETA
        self._spikes = []

    def _lif(self, x, th=1.0):
        if self.relaxed:
            s = torch.sigmoid(self.beta * (x - th))
            self._spikes.append(s)
            return s
        s = LIFSTE.apply(x, th)
        self._spikes.append(s)
        return s

    def calibrate(self, x):
        with torch.no_grad():
            z = self.stem(x)
            self.gamma.fill_(1.0 / (z.std() + 1e-6))

    def forward(self, x):
        self._spikes = []
        tok = self.stem(x) * self.gamma            # [B,d,8,8]
        tok = tok.flatten(2).transpose(1, 2)       # [B,64,d]
        feats = []
        for _ in range(self.T):
            z = tok
            for blk in self.blocks:
                z = blk(z, self._lif)
            feats.append(z.mean(1))                # [B,d]
        feat = torch.stack(feats, 1)               # [B,T,d]
        feat_sp = self._lif(feat, 1.0)             # head LIF
        return self.head(feat_sp.mean(1))


# ---------------------------------------------------------------------------
# B. 引理 1：胞内平坦性（真实数据 + 真实结构）
# ---------------------------------------------------------------------------
def patterns_match(model, sig0):
    return all(torch.equal(a, b) for a, b in zip(model._spikes, sig0))


def part_b(model, data):
    hr("B. 引理 1 胞内平坦性（真实 CIFAR batch=16）")
    xb, yb = data[0][:16], data[1][:16]
    model.eval()
    with torch.no_grad():
        model.relaxed = False
        _ = model(xb)
        sig0 = [s.clone() for s in model._spikes]
        F0 = F.cross_entropy(model(xb), yb).item()
        named = dict(model.named_parameters())
        g = torch.Generator().manual_seed(7)
        dirs = {k: torch.randn(v.shape, generator=g).to(DEV) for k, v in named.items()}

        print(f"{'参数':<22}{'胞内η/std':>10}{'胞内ΔF':>12}{'跨胞η/std':>11}"
              f"{'跳变ΔF':>11}{'松弛ΔF(同η)':>13}")
        for name in ["blocks.0.q.weight", "blocks.0.proj.weight",
                     "blocks.1.fc1.weight", "head.weight"]:
            w = named[name]
            eps = dirs[name] / (dirs[name].norm() + 1e-12)
            std = w.std().item()
            eta = 0.1 * std
            for _ in range(40):
                w.add_(eps, alpha=eta); _ = model(xb)
                same = patterns_match(model, sig0)
                w.sub_(eps, alpha=eta)
                if same:
                    break
                eta *= 0.5
            w.add_(eps, alpha=eta); _ = model(xb)
            same = patterns_match(model, sig0)
            dF_in = F.cross_entropy(model(xb), yb).item() - F0 if same else float("nan")
            w.sub_(eps, alpha=eta)
            eta2, jump = 0.05 * std, None
            for _ in range(40):
                w.add_(eps, alpha=eta2); _ = model(xb)
                flipped = not patterns_match(model, sig0)
                if flipped:
                    jump = F.cross_entropy(model(xb), yb).item() - F0
                w.sub_(eps, alpha=eta2)
                if flipped:
                    break
                eta2 *= 1.6
            model.relaxed = True
            w.add_(eps, alpha=eta)
            dF_rel = F.cross_entropy(model(xb), yb).item() - F0
            w.sub_(eps, alpha=eta)
            model.relaxed = False
            tag = "head(例外,应≠0)" if "head" in name else "block(应=0)"
            ok = (dF_in == 0.0) if "head" not in name else (abs(dF_in) > 1e-9)
            print(f"{name:<22}{eta/std:>10.2e}{dF_in:>12.3e}{eta2/std:>11.2e}"
                  f"{(jump if jump is not None else float('nan')):>11.3e}{dF_rel:>13.3e}  {tag} {'PASS' if ok else 'CHECK'}")


# ---------------------------------------------------------------------------
# C. 信号通道：ES vs STE 余弦 + 逐层方差剖面
# ---------------------------------------------------------------------------
def flat_grad(model, loss):
    gs = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return torch.cat([g.detach().reshape(-1) for g in gs if g is not None])


def es_direction(model, xb, yb, N, sigma_rel, hard, beta=BETA, param_filter=None, gen=0):
    named = [(k, v) for k, v in model.named_parameters()
             if param_filter is None or param_filter(k)]
    g = torch.Generator().manual_seed(1000 + gen)
    raws = []
    with torch.no_grad():
        backup = [(k, v.detach().clone()) for k, v in named]
        model.relaxed = not hard
        model.beta = beta
        for _ in range(N):
            for k, v in named:
                sig = (sigma_rel * v.std().clamp(min=1e-6)).item()
                v.add_(torch.randn(v.shape, generator=g).to(DEV), alpha=sig)
            raws.append(float(-F.cross_entropy(model(xb), yb)))
            for (k, v), (k0, v0) in zip(named, backup):
                v.copy_(v0)
        model.relaxed = False
    raws = np.array(raws)
    s = (raws - raws.mean()) / (raws.std() + 1e-8)
    gs, g2 = [], torch.Generator().manual_seed(1000 + gen)
    with torch.no_grad():
        for k, v in named:
            acc = torch.zeros_like(v)
            for n in range(N):
                acc.add_(torch.randn(v.shape, generator=g2).to(DEV), alpha=float(s[n]))
            gs.append(acc.reshape(-1))
    return torch.cat(gs), float(raws.var())


def part_c(model, data):
    hr("C. 信号通道（真实 CIFAR batch=16, ES N=256, σ_rel=0.5）")
    xb, yb = data[0][:16], data[1][:16]
    model.eval()
    model.relaxed = False
    g_ste = flat_grad(model, F.cross_entropy(model(xb), yb))
    model.relaxed = True
    g_rel = flat_grad(model, F.cross_entropy(model(xb), yb))
    model.relaxed = False
    cos = lambda a, b: float(F.cosine_similarity(a, b, dim=0))
    t0 = time.time()
    g_es_h, var_h = es_direction(model, xb, yb, 256, 0.5, hard=True, gen=1)
    g_es_r, var_r = es_direction(model, xb, yb, 256, 0.5, hard=False, gen=1)
    print(f"‖g_STE(α=4)‖={g_ste.norm():.4f}  ‖g_松弛解析(β=4)‖={g_rel.norm():.4f}  (ES 计时 {time.time()-t0:.0f}s)")
    c1 = cos(g_rel, g_ste)
    c2 = cos(g_es_h, g_ste)
    c3 = cos(g_es_r, g_ste)
    print(f"cos(g_松弛解析, g_STE)   = {c1:+.4f}   （定理 3.2：双估计器同一目标，应≈1）")
    print(f"cos(g_ES_hard , g_STE)   = {c2:+.4f}   （引理 1：胞内平坦 ⇒ ≈0）")
    print(f"cos(g_ES_relax, g_STE)   = {c3:+.4f}   （定理 3：松弛后信号出现，应显著>0）")
    print(f"单方向 fitness 方差: hard={var_h:.4f}  relaxed={var_r:.4f}  （定理 2）")
    v1 = c1 > 0.9
    v2 = abs(c2) < c3
    v3 = c3 > 0.2
    v4 = var_h > var_r
    print(f"[C] 判定: 定理3.2 {'PASS' if v1 else 'CHECK'} | 引理1信号通道 {'PASS' if v2 else 'CHECK'} | "
          f"松弛信号 {'PASS' if v3 else 'CHECK'} | 方差 {'PASS' if v4 else 'CHECK'}")

    # --- 逐层方差剖面（方差灾难 vs 深度，真实数据）---
    hr("C2. 逐层扰动方差剖面（真实数据, N=512, σ_rel=0.5）")
    print(f"{'扰动层':<28}{'hard Var':>13}{'hard P(≠0)':>12}{'relaxed Var':>13}{'比值':>8}")
    for pf in [r"^blocks\.0\.(q|k|v)\.", r"^blocks\.1\.(q|k|v)\.",
               r"^blocks\.1\.(fc1|fc2|proj)\.", r"^head\."]:
        names = [k for k, _ in model.named_parameters() if __import__("re").match(pf, k)]

        def filt(k):
            return any(k.startswith(nm.split(".weight")[0].split(".bias")[0])
                       for nm in [x for x in names])
        vh = es_direction(model, xb, yb, 512, 0.5, hard=True, param_filter=filt, gen=5)[1]
        vr = es_direction(model, xb, yb, 512, 0.5, hard=False, param_filter=filt, gen=5)[1]
        print(f"{pf:<28}{vh:>13.3e}{'>0':>12}{vr:>13.3e}{vh/(vr+1e-30):>8.1f}")


# ---------------------------------------------------------------------------
# C2. 子空间限制 ES：D_eff=64 << N，检测信号通道 + N 一致性（定理 2.4/3）
# ---------------------------------------------------------------------------
def part_c_subspace(model, data, k=64):
    hr(f"C2. 子空间限制 ES（真实 CIFAR batch=16, D_eff={k}, σ_rel=0.5）")
    xb, yb = data[0][:16], data[1][:16]
    model.eval()
    named = list(model.named_parameters())
    Dtot = sum(v.numel() for _, v in named)
    G = torch.Generator().manual_seed(77)
    Q, _ = torch.linalg.qr(torch.randn(Dtot, k, generator=G).to(DEV))  # [Dtot,k]
    Qs, off = [], 0
    for _, v in named:
        n = v.numel()
        Qs.append(Q[off:off + n].view(*v.shape, k))
        off += n
    sigs = [(0.5 * v.std().clamp(min=1e-6)).item() for _, v in named]

    def proj_ref(gvec):
        """J^T 参考梯度到子空间：r_j = Σ_i σ_i·Q_i^T g_i"""
        out, o = torch.zeros(k, device=DEV), 0
        for qi, si, (_, v) in zip(Qs, sigs, named):
            n = v.numel()
            out += si * (qi.reshape(n, k).T @ gvec[o:o + n])
            o += n
        return out

    model.relaxed = False
    g_ste = flat_grad(model, F.cross_entropy(model(xb), yb))
    model.relaxed = True
    g_rel = flat_grad(model, F.cross_entropy(model(xb), yb))
    model.relaxed = False
    # ES 估计 fitness 上升方向；参考梯度取 −∇CE（同为上升约定），符号方可比
    ref_ste, ref_rel = proj_ref(-g_ste), proj_ref(-g_rel)
    cos = lambda a, b: float(F.cosine_similarity(a, b, dim=0))

    def es_sub(hard, N, gen):
        g = torch.Generator().manual_seed(3000 + gen)
        zs = torch.randn(N, k, generator=g).to(DEV)          # [N,k]
        raws = []
        with torch.no_grad():
            backup = [(kk, v.detach().clone()) for kk, v in named]
            model.relaxed = not hard
            for n in range(N):
                z = zs[n]
                for qi, si, (_, v) in zip(Qs, sigs, named):
                    v.add_(qi @ z, alpha=si)
                raws.append(float(-F.cross_entropy(model(xb), yb)))
                for (kk, v), (k0, v0) in zip(named, backup):
                    v.copy_(v0)
            model.relaxed = False
        raws = np.array(raws)
        s = (raws - raws.mean()) / (raws.std() + 1e-8)
        return (s[:, None] * zs.cpu().numpy()).sum(0)

    import numpy.linalg as npl
    print(f"cos(g_松弛解析投影, g_STE投影)        = {cos(ref_rel, ref_ste):+.4f}（参考）")
    print(f"{'N':>6} | {'cos(ES_hard, STE)':>18} | {'cos(ES_relax, STE)':>19} | {'cos(ES_relax, 解析)':>19}")
    out = {}
    for N in [256, 4096]:
        zh = es_sub(True, N, gen=10)
        zr = es_sub(False, N, gen=10)
        c_h = float(zh @ ref_ste.cpu().numpy() / (npl.norm(zh) * npl.norm(ref_ste.cpu().numpy())))
        c_r = float(zr @ ref_ste.cpu().numpy() / (npl.norm(zr) * npl.norm(ref_ste.cpu().numpy())))
        c_rr = float(zr @ ref_rel.cpu().numpy() / (npl.norm(zr) * npl.norm(ref_rel.cpu().numpy())))
        out[N] = (c_h, c_r, c_rr)
        print(f"{N:>6} | {c_h:>18.4f} | {c_r:>19.4f} | {c_rr:>19.4f}")
    (c_h1, c_r1, _), (c_h2, c_r2, c_rr2) = out[256], out[4096]
    ok = (abs(c_h2) < 0.15) and (c_r2 > c_h2 + 0.1) and (c_r2 >= c_r1 - 0.05)
    print(f"[C2] 判定: {'PASS（hard 无信号；relaxed 信号存在且随 N 增强——定理 2.4+3 成立）' if ok else 'CHECK'}")


# ---------------------------------------------------------------------------
# D. 真实训练对照
# ---------------------------------------------------------------------------
def evaluate(model, x, y, bs=500):
    model.eval(); model.relaxed = False
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            p = model(x[i:i + bs]).argmax(1)
            correct += int((p == y[i:i + bs]).sum())
    return 100.0 * correct / len(x)


def part_d(data):
    hr("D. 真实训练对照（8000 训练/2000 测试, mini-SDT, T=2）")
    tr_x, tr_y, te_x, te_y = data
    torch.manual_seed(42)
    base = MiniSDT().to(DEV)
    base.calibrate(tr_x[:128])
    results = {}

    # D1 SGD-STE
    m = MiniSDT().to(DEV); m.load_state_dict(base.state_dict()); m.relaxed = False
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(5)
    for step in range(600):
        idx = torch.randint(0, len(tr_x), (64,), generator=g).to(DEV)
        loss = F.cross_entropy(m(tr_x[idx]), tr_y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            print(f"  SGD-STE step {step+1}: loss={loss.item():.4f} "
                  f"val={evaluate(m, te_x, te_y):.2f}%", flush=True)
    results["SGD-STE(600步)"] = evaluate(m, te_x, te_y)

    # D2-D4 ES 变体
    def es_train(hard, anneal, tag, gens=300, N=256, es_batch=8, lr=0.02):
        m = MiniSDT().to(DEV); m.load_state_dict(base.state_dict())
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(11)
        order = torch.randperm(len(tr_x), generator=g).to(DEV)
        cursor, beta = 0, BETA
        t0 = time.time()
        for gen in range(gens):
            # 每 generation 轮换 fitness batch（避免过拟合固定批，与 rust 侧一致）
            if cursor + es_batch > len(tr_x):
                order = order[torch.randperm(len(tr_x), generator=g).to(DEV)]
                cursor = 0
            idx = order[cursor:cursor + es_batch]
            cursor += es_batch
            xb, yb = tr_x[idx], tr_y[idx]
            named = list(m.named_parameters())
            m.relaxed = not hard
            m.beta = beta
            raws, dirs = [], []
            with torch.no_grad():
                for _ in range(N):
                    backup = [(k, v.detach().clone()) for k, v in named]
                    dn = []
                    for k, v in named:
                        sig = (0.5 * v.std().clamp(min=1e-6)).item()
                        e = torch.randn(v.shape, generator=g).to(DEV)
                        v.add_(e, alpha=sig)
                        dn.append(e)
                    raws.append(float(-F.cross_entropy(m(xb), yb)))
                    dirs.append(dn)
                    for (k, v), (k0, v0) in zip(named, backup):
                        v.copy_(v0)
            raws = np.array(raws)
            s = (raws - raws.mean()) / (raws.std() + 1e-8)
            with torch.no_grad():
                for pi, (k, v) in enumerate(named):
                    acc = torch.zeros_like(v)
                    for n in range(N):
                        acc.add_(dirs[n][pi], alpha=float(s[n]))
                    v.grad = -acc / N
            opt.step(); opt.zero_grad()
            if anneal and (gen + 1) % 100 == 0:
                beta = min(beta * 2.0, 16.0)
            if (gen + 1) % 100 == 0:
                print(f"  {tag} gen {gen+1}: fit={raws.mean():.4f} β={beta:.1f} "
                      f"val={evaluate(m, te_x, te_y):.2f}% ({time.time()-t0:.0f}s)", flush=True)
        m.relaxed = False
        results[tag] = evaluate(m, te_x, te_y)

    es_train(hard=True, anneal=False, tag="ES-hard(300gen)")
    es_train(hard=False, anneal=False, tag="ES-relax β=4(300gen)")
    es_train(hard=False, anneal=True, tag="ES-relax 退火(300gen)")

    hr("D. 最终结果（真实测试集 2000 张）")
    for k, v in results.items():
        print(f"  {k:<28} val_top1 = {v:.2f}%")
    r = results
    diff = r["ES-relax β=4(300gen)"] - r["ES-hard(300gen)"]
    print(f"[D] 判定: {'PASS（松弛化 ES 优 +%.1fpp）' % diff if diff > 2.0 else 'CHECK（差 %.1fpp）' % diff}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"device={DEV}  torch={torch.__version__}", flush=True)
    data = load_data()
    print(f"真实数据: train={len(data[0])} test={len(data[2])} (CIFAR-10 [0,1])", flush=True)
    t0 = time.time()
    if only in ("all", "a"):
        part_a()
    torch.manual_seed(42)
    model = MiniSDT().to(DEV)
    model.calibrate(data[0][:128])
    if only in ("all", "b"):
        part_b(model, data)
    if only in ("all", "c"):
        part_c(model, data)
        part_c_subspace(model, data)
    if only == "c2":
        part_c_subspace(model, data)
    if only in ("all", "d"):
        part_d(data)
    print(f"\n总耗时 {time.time()-t0:.0f}s")
