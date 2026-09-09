# -*- coding: utf-8 -*-
"""T5: 松弛谱系对比 — 谁的梯度方向最接近真平滑景观 grad F_sigma?
基准: 精确混合景观 (2^H 状态枚举, 解析, 无 MC) — ES MC 应与之互证
候选: (a) 均值场 probit (统一 beta)  (b) 均值场 + 逐层 beta  (c) 方差传播 VP
判据: cos(候选, grad F_exact) > 0.95 → 该松弛作为 SGD 侧实现作用等价
"""
import numpy as np
from math import erf
from itertools import product

rng = np.random.default_rng(11)

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

def Phi(u):
    return 0.5 * (1 + np.vectorize(erf)(u / np.sqrt(2)))

d, H, B = 8, 4, 32
X = rng.normal(size=(B, d))
X /= np.linalg.norm(X, axis=1, keepdims=True) / 2.0
nx1 = np.linalg.norm(X, axis=1)          # ||x_i||
ts = rng.integers(0, 2, B).astype(float)
sigma = 0.7
STATES = np.array(list(product([0, 1], repeat=H)))   # [2^H, H]
SN = np.sqrt(STATES.sum(axis=1) + 1)                 # 每状态外层噪声宽度因子

def init_params(seed):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 0.5, (H, d))
    v1 = (X @ W1.T).mean(axis=0) + r.normal(0, 0.2, H)
    return np.concatenate([W1.reshape(-1), v1, r.normal(1.0, 0.2, H), [0.5]])

def unpack(p):
    return p[:H * d].reshape(H, d), p[H * d:H * d + H], p[H * d + H:H * d + 2 * H], p[-1]

def d1_matrix(p):
    W1, v1, _, _ = unpack(p)
    return X @ W1.T - v1                                  # [B,H] 膜距

def q1_matrix(p):
    return Phi(-d1_matrix(p) / (sigma * np.sqrt(nx1 ** 2 + 1)[:, None]))  # [B,H] 翻转概率

def hard_loss(p):
    W1, v1, w2, v2 = unpack(p)
    S = (X @ W1.T > v1).astype(float)
    y = (S @ w2 > v2).astype(float)
    return np.mean((1 - 2 * ts) * y + ts ** 2)

def grad_F_exact(p):
    """精确混合景观梯度 (状态枚举): 内层基态 s0 (ON/OFF 混合), 翻转概率对两侧对称"""
    W1, v1, w2, v2 = unpack(p)
    D1 = d1_matrix(p)                                     # [B,H]
    w_h = (sigma * np.sqrt(nx1 ** 2 + 1))[:, None]        # [B,1]
    Q = Phi(-np.abs(D1) / w_h)                            # [B,H] 翻转概率 (两侧对称)
    S0 = (D1 > 0).astype(float)                           # 内层基态
    g = np.zeros_like(p)
    Pw = np.zeros((B, len(STATES)))
    Out = np.zeros((B, len(STATES)))
    for si, s in enumerate(STATES):
        agree = (s == S0)                                 # [B,H] 与基态一致?
        Pw[:, si] = np.prod(np.where(agree, 1 - Q, Q), axis=1)
        Out[:, si] = Phi((s @ w2 - v2) / (sigma * SN[si]))
    Fterms = (1 - 2 * ts)[:, None] * Out + ts[:, None] ** 2        # [B, 2^H]
    phiOut = phi((STATES @ w2 - v2)[None, :] / (sigma * SN)[None, :]) / (sigma * SN)[None, :]
    Wgt = (1 - 2 * ts)[:, None] * Pw * phiOut                       # [B, 2^H]
    g[H * d + H:H * d + 2 * H] = np.mean(Wgt @ STATES, axis=0)
    g[-1] = -np.mean(Wgt.sum(axis=1))
    # dF/dq_h = sum_s dP/dq_h * Fterms ; dP/dq_h = P * (不一致? +1/(q) : -q/(1-q) 符号)
    # 翻转概率 q: P(s) 中 "翻转到位" 的因子是 q (不一致), "保持" 是 (1-q) (一致)
    for h in range(H):
        flip = (STATES[None, :, h] != S0[:, h][:, None]).astype(float)   # [B, 2^H]
        dPdq = Pw * np.where(flip == 1, 1.0 / np.maximum(Q[:, h][:, None], 1e-9),
                             -1.0 / np.maximum(1 - Q[:, h][:, None], 1e-9))
        dFdq = np.sum(dPdq * Fterms, axis=1)                    # [B]
        # dq/dW1_h: |d1| 减小 → q 增大: dq/dW1 = -sign(d1)*phi(|d1|/w)/w * x
        dq = -np.sign(D1[:, h]) * phi(np.abs(D1[:, h]) / w_h[:, 0]) / w_h[:, 0]   # [B]
        g[h * d:(h + 1) * d] += np.mean(dFdq[:, None] * dq[:, None] * X, axis=0)
        g[H * d + h] += np.mean(dFdq * dq * (-1.0))
    return g

def grad_meanfield(p, per_layer=True):
    W1, v1, w2, v2 = unpack(p)
    b1 = 1.0 / (sigma * np.sqrt(2.0 ** 2 + 1.0)) if per_layer else 1.0 / sigma
    b2 = 1.0 / sigma
    h1 = X @ W1.T - v1
    s1 = Phi(b1 * h1)
    m2 = s1 @ w2 - v2
    c = (1 - 2 * ts) * phi(b2 * m2) * b2
    gh1 = c[:, None] * w2[None, :] * phi(b1 * h1) * b1
    g = np.zeros_like(p)
    g[:H * d] = (gh1[:, :, None] * X[:, None, :]).mean(axis=0).reshape(-1)
    g[H * d:H * d + H] = (gh1 * (-1)).mean(axis=0)
    g[H * d + H:H * d + 2 * H] = (c[:, None] * s1).mean(axis=0)
    g[-1] = (c * (-1)).mean()
    return g

def grad_vp(p):
    """方差传播松弛: 外层膜 ~ N(mu, var), mu = w2.s1~ - v2,
    var = sum_h w2_h^2 q(1-q) + sigma^2*(||s1~||_1 + 1)  (矩匹配)"""
    W1, v1, w2, v2 = unpack(p)
    b1 = 1.0 / (sigma * np.sqrt(2.0 ** 2 + 1.0))
    h1 = X @ W1.T - v1
    Q = Phi(-h1 / (sigma * np.sqrt(nx1 ** 2 + 1)[:, None]))    # 翻转概率 [B,H]
    s1 = 1 - Q                                                  # E[s1]
    mu = s1 @ w2 - v2                                           # [B]
    var = (w2 ** 2 * Q * (1 - Q)).sum(axis=1) + sigma ** 2 * (s1.sum(axis=1) + 1)
    sd = np.sqrt(var)
    u = mu / sd
    c0 = (1 - 2 * ts) * phi(u)                                  # [B]
    g = np.zeros_like(p)
    # dPhi/dmu = phi(u)/sd ; dPhi/dsd = phi(u)*(-mu/sd^2)  — 链式: phi(u) 只乘一次
    dmu = 1.0 / sd
    dsd = -mu / sd ** 2
    # w2: dmu/dw2 = s1 ; dvar/dw2_h = 2 w2_h q_h (1-q_h)
    g[H * d + H:H * d + 2 * H] = np.mean(c0[:, None] * (dmu[:, None] * s1 + dsd[:, None] * (2 * w2[None, :] * Q * (1 - Q))), axis=0)
    g[-1] = np.mean(c0 * (dmu * (-1) + dsd * (sigma ** 2 / (2 * sd))))
    # v1, W1: 通过 Q (s1 = 1-Q, var 项也含 Q)
    dq_dw1 = -phi(h1 / (sigma * np.sqrt(nx1 ** 2 + 1)[:, None])) / (sigma * np.sqrt(nx1 ** 2 + 1)[:, None])  # [B,H]
    for h in range(H):
        dmu_dq = -w2[h]
        dvar_dq = w2[h] ** 2 * (1 - 2 * Q[:, h]) + sigma ** 2 * (-1)
        dsd_dq = dvar_dq / (2 * sd)
        c_h = c0 * (dmu * dmu_dq + dsd * dsd_dq)                # [B] dL/dQ
        g[h * d:(h + 1) * d] = np.mean(c_h[:, None] * dq_dw1[:, h][:, None] * X, axis=0) * (-1) * 0 + \
                               np.mean((-c_h[:, None]) * (-dq_dw1[:, h][:, None]) * X, axis=0) * 0
        # dL/dW1_h = dL/dQ * dQ/dW1_h ; dQ/dW1_h = -phi(...)*x/(...)
        g[h * d:(h + 1) * d] += np.mean(c_h[:, None] * dq_dw1[:, h][:, None] * X, axis=0)
        g[H * d + h] += np.mean(c_h * dq_dw1[:, h] * (-1.0))
    return g

print("=== 互证: ES MC vs 精确混合解析 (cos 应 ~1) ===")
p0 = init_params(0)
gF = grad_F_exact(p0)
def es_mc(p, N=120000):
    E = rng.normal(size=(N, len(p)))
    fs = np.array([hard_loss(p + sigma * e) for e in E])
    return ((fs - fs.mean())[:, None] * E).mean(axis=0) / sigma
gmc = es_mc(p0)
print(f"cos(ES_MC, gradF_exact) = {gmc @ gF / (np.linalg.norm(gmc) * np.linalg.norm(gF)):.4f}  |gF|={np.linalg.norm(gF):.4f}")

print("=== 松弛谱系 vs 精确混合 (3 个轨迹点) ===")
p = p0
for step in [0, 50, 99]:
    gF = grad_F_exact(p)
    gm = grad_meanfield(p, per_layer=True)
    gv = grad_vp(p)
    cf = gm @ gF / (np.linalg.norm(gm) * np.linalg.norm(gF))
    cv = gv @ gF / (np.linalg.norm(gv) * np.linalg.norm(gF))
    print(f"step {step:>3}: cos(均值场+逐层β) = {cf:.4f}   cos(方差传播VP) = {cv:.4f}   |gF|={np.linalg.norm(gF):.4f}")
    p = p - 0.1 * grad_meanfield(p, per_layer=True)
