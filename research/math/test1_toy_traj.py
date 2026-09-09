# -*- coding: utf-8 -*-
"""小批量测试 1 (v2): 分离两个问题
Phase A 方向对齐: 沿 SGD 轨迹的每一点 p_t, 同时计算 grad_L~_probit(p_t) 与
         ES 估计器方向(p_t, pop=4096 反反对称, 逐步 k 重校准) — cos 是
         纯估计器对齐度 (排除轨迹分岔混杂)
Phase B 独立轨迹: SGD 与 ES 各自训练 3 种子, 比较损失曲线 (均值±散布)
判据: Phase A 平均 cos > 0.95; Phase B 损失曲线统计重合
"""
import numpy as np

rng = np.random.default_rng(11)

def Phi(u):
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(u / np.sqrt(2)))

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

d, H, B = 8, 4, 32
X = rng.normal(size=(B, d))
X /= np.linalg.norm(X, axis=1, keepdims=True) / 2.0
ts = rng.integers(0, 2, B).astype(float)

def init_params(seed):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 0.5, (H, d))
    v1 = (X @ W1.T).mean(axis=0) + r.normal(0, 0.2, H)
    return np.concatenate([W1.reshape(-1), v1, r.normal(1.0, 0.2, H), [0.5]])

def unpack(p):
    return p[:H * d].reshape(H, d), p[H * d:H * d + H], p[H * d + H:H * d + 2 * H], p[-1]

def hard_loss(p):
    W1, v1, w2, v2 = unpack(p)
    S = (X @ W1.T > v1).astype(float)
    y = (S @ w2 > v2).astype(float)
    return np.mean((1 - 2 * ts) * y + ts ** 2)

beta = 1.0 / 0.7
sigma = 0.7
pop = 4096
n_steps = 100
eta = 0.1

def sgd_grad(p):
    W1, v1, w2, v2 = unpack(p)
    h1 = X @ W1.T - v1
    s1 = Phi(beta * h1)
    m2 = s1 @ w2 - v2
    c = (1 - 2 * ts) * phi(beta * m2) * beta
    gh1 = c[:, None] * w2[None, :] * phi(beta * h1)
    g = np.zeros_like(p)
    g[:H * d] = (gh1[:, :, None] * X[:, None, :]).mean(axis=0).reshape(-1)
    g[H * d:H * d + H] = (gh1 * (-1)).mean(axis=0)
    g[H * d + H:H * d + 2 * H] = (c[:, None] * s1).mean(axis=0)
    g[-1] = (c * (-1)).mean()
    return g

def es_dir(p, rs):
    """ES 估计器方向: E[sum(z*eps)/sqrt(pop)] = +k*grad(F); k 逐步重校准"""
    E = rs.normal(size=(pop // 2, len(p)))
    E = np.concatenate([E, -E], axis=0)
    fs = np.array([hard_loss(p + sigma * e) for e in E])
    std_f = fs.std()
    z = (fs - fs.mean()) / std_f
    g_dir = (z[:, None] * E).sum(axis=0) / np.sqrt(pop)
    k = np.sqrt(pop) * sigma / std_f
    return g_dir, k

print("=== Phase A: 同点方向对齐 (沿 SGD 轨迹) ===")
p = init_params(0)
cos_list, alive = [], []
for step in range(n_steps):
    g_sgd = sgd_grad(p)
    g_dir, k = es_dir(p, rng)
    nrm = np.linalg.norm(g_sgd) * np.linalg.norm(g_dir) + 1e-12
    cos_list.append(g_sgd @ g_dir / nrm)
    alive.append(np.linalg.norm(g_sgd) > 0.05)   # 信号活着 (非平台)
    p = p - eta * g_sgd
cos_arr = np.array(cos_list)
alive = np.array(alive)
print(f"全部步:   平均 cos = {cos_arr.mean():.4f}  最小 = {cos_arr.min():.4f}")
print(f"信号活着: 平均 cos = {cos_arr[alive].mean():.4f}  最小 = {cos_arr[alive].min():.4f}  ({alive.sum()}/{n_steps} 步)")

print("=== Phase B: 独立轨迹损失曲线 (3 种子) ===")
for seed in range(3):
    ps = init_params(seed)
    pe = init_params(seed)
    curve = []
    for step in range(n_steps):
        curve.append((hard_loss(ps), hard_loss(pe)))
        ps = ps - eta * sgd_grad(ps)
        g_dir, k = es_dir(pe, np.random.default_rng(100 + seed * 1000 + step))
        pe = pe - eta * g_dir / k
    curve = np.array(curve)
    print(f"seed{seed}: SGD L: {curve[0,0]:.3f}→{curve[-1,0]:.3f}   ES L: {curve[0,1]:.3f}→{curve[-1,1]:.3f}   "
          f"末段均值 SGD={curve[-20:,0].mean():.3f} ES={curve[-20:,1].mean():.3f}")
