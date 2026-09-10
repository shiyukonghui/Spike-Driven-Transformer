# -*- coding: utf-8 -*-
"""Round 6: 相关传播松弛 (CP) — T5 的宽度空间推广, 真实第一层
ES 位移相关结构: delta_h = sigma*M*eps, K = M M^T (秩 928, 稀疏耦合)
CP 松弛: 均值 = 逐边界 probit 边缘; 读出方差 = u^T Cov[s] u
  Cov[1_f,1_g] ~= rho_fg * phi_f * phi_g  (copula 一阶)
  u^T K u 分解: A=u*phi/||a||;  A^T K A = ||P^T S||^2 项 + sum_c (sum_p A_cp)^2 项
损失: R6 精确二次单读出  L = (1-2t)*1[sum_f u_f s_f > 0] + t^2
对比: (a) 逐位置 probit 基线  (b) CP  — vs ES 硬 MC (N=4M x2, 信度校正)
"""
import numpy as np
import torch
import torch.nn.functional as tF
torch.manual_seed(0)
rng = np.random.default_rng(0)
device = 'cuda'
z = np.load(r'artifacts\mnist\cifar10_data.npz')
xt, yt = z['train_x'], z['train_y']
B = 128
idx = rng.choice(xt.shape[0], B, replace=False)
x = torch.tensor(xt[idx], dtype=torch.float32, device=device)
lab = torch.tensor(yt[idx], dtype=torch.long, device=device)
zw = np.load(r'artifacts\train_burn_init.npz')
w = torch.tensor(zw['pe_proj0_w'].astype(np.float32), requires_grad=True, device=device)
b = torch.tensor(zw['pe_proj0_b'].astype(np.float32), requires_grad=True, device=device)
thr = 1.0
F2 = 32 * 32  # 1024 位置

# R6 二次读出: u [32768] 固定随机, 类 k=3 的 one-vs-rest
torch.manual_seed(7)
probe = torch.randn(32 * 32 * 32, 10, device=device) * 0.02
k = 3
u = probe[:, k].detach().clone()                      # [F]
t_bin = (lab == k).float()                            # [B]
print(f"readout: 正类率 = {t_bin.mean():.3f}, |u| = {u.norm():.3f}", flush=True)

# 预计算: patch 矩阵 Pmat [1024, 27], a_norm [B, 1024]
patches = tF.unfold(x, kernel_size=3, padding=1).transpose(1, 2)   # [B, 1024, 27]
Pmat = patches[0].contiguous()                          # [1024, 27] (patch 向量, 每样本相同? 不 — 每样本自己的 patch)
def a_norm_of(xx):
    return (xx.reshape(B, 3, F2).norm(dim=1) ** 2 + 1).sqrt()      # [B, 1024]

def phi(t):
    return torch.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)

def grad_relax(mode, sigma):
    """mode: 'probit' 逐位置 probit; 'cp' 相关传播"""
    if w.grad is not None: w.grad = None
    if b.grad is not None: b.grad = None
    h = tF.conv2d(x, w, b, stride=1, padding=1)                     # [B,32,32,32]
    an = a_norm_of(x).reshape(B, 1, 32, 32)                         # [B,1,32,32]
    wf = sigma * an
    zf = (h - thr) / wf
    Pf = 0.5 * (1 + torch.erf(zf / np.sqrt(2)))
    ph = phi(zf)
    U = u.reshape(1, 32, 32, 32)
    mu = (U * Pf).flatten(1).sum(1)                                 # [B] 读出均值
    if mode == 'probit':
        v = (U ** 2 * Pf * (1 - Pf)).flatten(1).sum(1) + 1e-12
    else:
        # CP: mu 同上; v = diag(P(1-P)) + A^T K A - diag(phi^2)
        # K[(c,p),(c',p')] = x_{i,p}.x_{i,p'} + delta_cc'  (逐样本 patch)
        A2 = (U * ph / an).reshape(B, 32, F2)                       # [B, 32, 1024]
        term2 = ((A2.sum(dim=2)) ** 2).sum(dim=1)                   # sum_c (sum_p A_cp)^2
        S = A2.sum(dim=1)                                           # [B, 1024] sum_c A_cp
        term1 = (torch.einsum('bpd,bp->bd', patches, S) ** 2).sum(dim=1)  # ||sum_p S_p x_p||^2
        diag_ph2 = (U ** 2 * ph ** 2).flatten(1).sum(1)
        v = (U ** 2 * Pf * (1 - Pf)).flatten(1).sum(1) + term1 + term2 - diag_ph2 + 1e-12
    r = mu / torch.sqrt(v)
    loss = ((1 - 2 * t_bin) * 0.5 * (1 + torch.erf(r / np.sqrt(2))) + t_bin ** 2).mean()
    loss.backward()
    return torch.cat([w.grad.flatten(), b.grad.flatten()])

# ES 硬 MC: 同一二读出损失
dim = w.numel() + b.numel()
def grad_es(sigma, N=4_000_000):
    M2 = N // 2
    g_acc = torch.zeros(dim, device=device)
    CH = 256
    done = 0
    with torch.no_grad():
        while done < M2:
            cc = min(CH, M2 - done)
            Ec = torch.randn(cc, dim, device=device)
            acc = torch.zeros(cc, device=device)
            for sgn in (+1.0, -1.0):
                Pw = w[None] + sgn * sigma * Ec[:, :w.numel()].reshape(-1, *w.shape)
                Pb = b[None] + sgn * sigma * Ec[:, w.numel():]
                Wb = Pw.repeat_interleave(B, dim=0).reshape(cc * B * 32, 3, 3, 3)
                bb = Pb.repeat_interleave(B, dim=0).reshape(-1)
                xin = x[None].expand(cc, -1, -1, -1, -1).reshape(1, cc * B * 3, 32, 32)
                h = tF.conv2d(xin, Wb, bb, stride=1, padding=1, groups=cc * B)
                h = h.reshape(cc, B, 32, 32, 32)
                s = (h > thr).float().reshape(cc * B, 32 * 32 * 32)
                ro = (s @ u).reshape(cc, B)                          # 读出和
                y = (ro > 0).float()
                ce = (1 - 2 * t_bin[None, :]) * y + t_bin[None, :] ** 2
                acc += sgn * ce.mean(1)
            g_acc += torch.einsum('c,cd->d', acc, Ec)
            done += cc
    return g_acc / (N * sigma)

def cos(a, b): return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

g_pro = grad_relax('probit', 0.10)
g_cp = grad_relax('cp', 0.10)
print(f"|g_probit|={g_pro.norm():.5f}  |g_cp|={g_cp.norm():.5f}", flush=True)
e1 = grad_es(0.10)
e2 = grad_es(0.10)
r = cos(e1, e2)
print(f"ES reliability = {r:.4f}", flush=True)
print(f"probit 逐位置 (quad loss): raw cos = {cos(g_pro, e1):.4f}  disatt = {cos(g_pro,e1)/np.sqrt(max(r,1e-9)):.4f}")
print(f"CP 相关传播 (quad loss):   raw cos = {cos(g_cp, e1):.4f}  disatt = {cos(g_cp,e1)/np.sqrt(max(r,1e-9)):.4f}")
