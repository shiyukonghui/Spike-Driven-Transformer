# -*- coding: utf-8 -*-
"""SGD-ES 作用等价: 形式化定理验证 (T1 景观恒等 / T2 更新率匹配 / T3 深度平均场间隙)
T1: 单 spike 层, probit 代理 beta=1/(sigma*||x||):  L~_beta(w) = F_sigma(w) 逐位恒等
    多样本时恒等条件 = 范数齐次 (sigma*||x_i|| = 1/beta ∀i)
T2: 管线估计器 E[g_pipe] = -k * grad F_sigma (k = 常数尺度) -> eta_ES = eta_SGD/k 等价更新
T3: 两层复合: F_sigma = 混合 (内层翻转概率 q1 加权), L~_beta = 平均场 (内层均值代入)
    间隙 = 内层方差项; q1->0 (内层死区) 时间隙->0 — 等价与信号的张力的数值刻画
"""
import numpy as np
from math import erf

rng = np.random.default_rng(5)

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

def Phi(u):
    return 0.5 * (1 + np.vectorize(erf)(u / np.sqrt(2)))

print("=== T1a: 单样本景观恒等 (beta = 1/(sigma*||x||)) ===")
d = 6
x = rng.normal(size=d)
nx = np.linalg.norm(x)
t = 1.0
sigma = 0.7
beta = 1.0 / (sigma * nx)
v = 0.3
wx_grid = np.linspace(-3, 3, 601)
# F_sigma(w) 沿 w = (wx/nx^2) x 方向: E_eps[(1-2t)*1[wx + sigma*eps.x > v] + t^2]
F_sigma = (1 - 2 * t) * Phi((wx_grid - v) / (sigma * nx)) + t ** 2
# L~_beta (probit 松弛, 线性化损失 R6): (1-2t)*Phi(beta*(wx - v)) + t^2
L_relax = (1 - 2 * t) * Phi(beta * (wx_grid - v)) + t ** 2
print(f"max |L~_beta - F_sigma| = {np.abs(L_relax - F_sigma).max():.2e}  (0 = 恒等)")

print("=== T1b: 多样本范数非齐次的间隙 ===")
norms = np.array([1.0, 2.0, 4.0])
ts4 = np.array([1.0, 0.0, 1.0])
for beta_try, tag in [(1.0 / (sigma * 2.0), "beta=1/(sigma*mean_norm)"), (2.0, "beta=2 (任选)")]:
    gaps = []
    for wx in wx_grid[::12]:
        F = np.mean((1 - 2 * ts4) * Phi((wx * norms / norms - v) / (sigma * norms)) + ts4 ** 2)
        # 每样本 w.x_i = wx (沿各自 x_i 方向比较) -> F = mean (1-2t)Phi((wx-v)/(sigma*||x_i||))+t^2
        F = np.mean((1 - 2 * ts4) * Phi((wx - v) / (sigma * norms)) + ts4 ** 2)
        L = np.mean((1 - 2 * ts4) * Phi(beta_try * (wx - v)) + ts4 ** 2)
        gaps.append(abs(F - L))
    print(f"{tag}: max gap = {max(gaps):.4f} (范数越散, 间隙越大)")

print("=== T1c: 范数齐次恢复恒等 ===")
norms_h = np.array([2.0, 2.0, 2.0])
gaps_h = []
for wx in wx_grid[::12]:
    F = np.mean((1 - 2 * ts4) * Phi((wx - v) / (sigma * norms_h)) + ts4 ** 2)
    L = np.mean((1 - 2 * ts4) * Phi((1.0 / (sigma * 2.0)) * (wx - v)) + ts4 ** 2)
    gaps_h.append(abs(F - L))
print(f"齐次范数 max gap = {max(gaps_h):.2e}")

print("=== T2: 管线估计器的尺度常数 (z-score + 1/sqrt(pop) 归一) ===")
# 单层 pop 个候选: 候选 c 的 fitness f_c = L_hard(w_c), w_c = w + sigma*eps_c
# 管线: g_pipe = sum_c z_c * eps_c / sqrt(pop),  z_c = (f_c - mean)/std
# ES 平滑景观梯度 grad F = (1/sigma) E[(f - mean) eps]  (R2/R3)
# 断言: E[g_pipe] = -k * grad F_sigma, k = ?  (数值测出, 检查与 theta 无关)
w_true = rng.normal(size=d) * 0.5
def L_hard_1(w):
    return (1 - 2 * t) * float(w @ x > v) + t ** 2
def grad_F_sigma(w):
    z_ = (w @ x - v) / (sigma * nx)
    return -(1 - 2 * t) * phi(z_) * x / (sigma * nx) / 1.0
pop = 20000
for wt in [w_true, w_true * 0.5, rng.normal(size=d) * 0.3]:
    E = rng.normal(size=(pop, d))
    fs = np.array([L_hard_1(wt + sigma * e) for e in E])
    z = (fs - fs.mean()) / fs.std()
    g_pipe = (z[:, None] * E).sum(axis=0) / np.sqrt(pop)
    gf = grad_F_sigma(wt)
    k = -np.linalg.norm(g_pipe) / np.linalg.norm(gf) * np.sign(g_pipe @ gf)
    print(f"|g_pipe|={np.linalg.norm(g_pipe):.5f} |gradF|={np.linalg.norm(gf):.5f} 尺度 k={k:.4f}")
print("理论: E[g_pipe] = -sqrt(pop)*phi((wx-v)/s)/(sigma*std_f) * dx方向 ... k 应为常数 (与 w 无关)")

print("=== T3: 两层复合的平均场间隙 (C6 张力) ===")
# 内层: s1 = 1[w1.x - v1 > 0], 膜距 d1; 外层: y = 1[w2*s1 - v2 > 0], t 给定
# F_sigma 解析: P(m>0) = (1-q1)*Phi((w2-v2)/sigma) + q1*1[v2<0],  q1 = Phi(-d1/(sigma*sqrt(nx^2+1)))
# L~_beta: (1-2t)*Phi(beta*(w2*Phi(beta*d1) - v2)) + t^2
w2, v2 = 1.5, 0.5
q_out = Phi((w2 - v2) / sigma)  # 外层 ON 时翻转.. 实际 P(m>0|s1=1)
print(f"{'d1':>6} {'q1(内层翻转)':>12} {'F_sigma':>10} {'L~_beta':>10} {'间隙':>10}")
beta2 = 1.0 / sigma  # 外层宽度对齐
for d1 in [0.2, 0.5, 1.0, 2.0, 3.0]:
    q1 = Phi(-d1 / (sigma * np.sqrt(nx ** 2 + 1)))
    F2 = (1 - 2 * t) * ((1 - q1) * Phi((w2 - v2) / sigma) + q1 * float(v2 < 0)) + t ** 2
    L2 = (1 - 2 * t) * Phi(beta2 * (w2 * Phi(beta2 * d1) - v2)) + t ** 2
    print(f"{d1:>6.1f} {q1:>12.4f} {F2:>10.5f} {L2:>10.5f} {abs(F2-L2):>10.5f}")
print("预期: d1 大 (内层死区, q1->0) 间隙->0; d1 小 (信号区, q1 大) 间隙开 — 等价与信号的张力")
