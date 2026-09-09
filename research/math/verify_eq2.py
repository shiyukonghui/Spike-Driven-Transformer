# -*- coding: utf-8 -*-
"""T4: 深度方向对齐定理 — cos(grad L~_probit, grad F_sigma) 随内层膜距/深度/范数异质
T4a: 两层, 齐次范数: cos vs d1 (内层膜距) — SGD 解析链 vs ES MC
T4b: 三层: 同上 (ES 侧 MC 硬网络)
T4c: 范数异质多样本: cos 随范数散布
结论判据: cos > 0.95 = 方向等价成立 (P2 路径可行)
"""
import numpy as np

rng = np.random.default_rng(7)

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

def Phi(u):
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(u / np.sqrt(2)))

d = 6
x = rng.normal(size=d)
nx = np.linalg.norm(x)
t = 1.0
sigma = 0.7
beta = 1.0 / sigma  # 外层宽度对齐 (x 方向归一后 ||x|| 项并入)

def grad_L_analytic(w1, v1, w2, v2, layers=2, w3=1.5, v3=0.5):
    """probit 松弛网络解析梯度 (线性化损失), layers=2 或 3"""
    s1 = Phi(beta * (w1 @ x - v1))
    if layers == 2:
        m2 = w2 * s1 - v2
        g = np.zeros(d + 3)
        g[:d] = (1 - 2 * t) * phi(beta * m2) * beta * w2 * phi(beta * (w1 @ x - v1)) * beta * x
        g[d] = -(1 - 2 * t) * phi(beta * m2) * beta * w2 * phi(beta * (w1 @ x - v1)) * beta
        g[d + 1] = (1 - 2 * t) * phi(beta * m2) * beta * s1
        g[d + 2] = -(1 - 2 * t) * phi(beta * m2) * beta
        return g
    else:  # 3 层: w3 标量在 s2 上
        s2 = Phi(beta * (w2 * s1 - v2))
        m3 = w3 * s2 - v3
        g = np.zeros(d + 5)
        common = (1 - 2 * t) * phi(beta * m3) * beta
        g[:d] = common * w3 * phi(beta * (w2 * s1 - v2)) * beta * w2 * phi(beta * (w1 @ x - v1)) * beta * x
        g[d] = -common * w3 * phi(beta * (w2 * s1 - v2)) * beta * w2 * phi(beta * (w1 @ x - v1)) * beta
        g[d + 1] = common * w3 * phi(beta * (w2 * s1 - v2)) * beta * s1
        g[d + 2] = -common * w3 * phi(beta * (w2 * s1 - v2)) * beta
        g[d + 3] = common * s2
        g[d + 4] = -common
        return g

def grad_F_mc(theta, layers=2, N=150000):
    """ES 平滑景观梯度: 硬网络 MC"""
    dim = len(theta)
    E = rng.normal(size=(N, dim))
    fs = np.empty(N)
    w1 = theta[:d]; v1 = theta[d]; w2 = theta[d + 1]; v2 = theta[d + 2]
    if layers == 3:
        w3 = theta[d + 3]; v3 = theta[d + 4]
    for i in range(N):
        e = E[i]
        w1p = w1 + sigma * e[:d]; v1p = v1 + sigma * e[d]
        s1 = float(w1p @ x > v1p)
        if layers == 2:
            w2p = w2 + sigma * e[d + 1]; v2p = v2 + sigma * e[d + 2]
            y = float(w2p * s1 > v2p)
        else:
            w2p = w2 + sigma * e[d + 1]; v2p = v2 + sigma * e[d + 2]
            s2 = float(w2p * s1 > v2p)
            w3p = w3 + sigma * e[d + 3]; v3p = v3 + sigma * e[d + 4]
            y = float(w3p * s2 > v3p)
        fs[i] = (1 - 2 * t) * y + t ** 2
    return ((fs - fs.mean())[:, None] * E).mean(axis=0) / sigma

print("=== T4a: 两层 cos vs 内层膜距 d1 (齐次范数) ===")
w2_0, v2_0 = 1.5, 0.5
w1_base = x / nx  # w1.x = 1
print(f"{'d1':>6} {'cos':>8} {'|gL|':>9} {'|gF|':>9}")
for d1 in [0.2, 0.5, 1.0, 2.0, 3.0]:
    v1 = w1_base @ x - d1
    theta = np.concatenate([w1_base, [v1, w2_0, v2_0]])
    gL = grad_L_analytic(w1_base, v1, w2_0, v2_0, layers=2)
    gF = grad_F_mc(theta, layers=2)
    cos = gL @ gF / np.linalg.norm(gL) / np.linalg.norm(gF)
    print(f"{d1:>6.1f} {cos:>8.4f} {np.linalg.norm(gL):>9.4f} {np.linalg.norm(gF):>9.4f}")

print("=== T4b: 三层 cos vs d1 ===")
w3_0, v3_0 = 1.5, 0.5
for d1 in [0.2, 0.5, 1.0, 2.0, 3.0]:
    v1 = w1_base @ x - d1
    theta = np.concatenate([w1_base, [v1, w2_0, v2_0, w3_0, v3_0]])
    gL = grad_L_analytic(w1_base, v1, w2_0, v2_0, layers=3)
    gF = grad_F_mc(theta, layers=3)
    cos = gL @ gF / np.linalg.norm(gL) / np.linalg.norm(gF)
    print(f"{d1:>6.1f} {cos:>8.4f}")

print("=== T4c: 范数异质多样本 (3 样本, 范数 1/2/4) ===")
norms = np.array([1.0, 2.0, 4.0])
xs = np.array([x * n / nx for n in norms])
ts3 = np.array([1.0, 0.0, 1.0])
def grad_L_samples(w1, v1, w2, v2):
    g = np.zeros(d + 3)
    for xi, ti in zip(xs, ts3):
        s1 = Phi(beta * (w1 @ xi - v1))
        m2 = w2 * s1 - v2
        c = (1 - 2 * ti) * phi(beta * m2) * beta
        g[:d] += c * w2 * phi(beta * (w1 @ xi - v1)) * beta * xi
        g[d] += -c * w2 * phi(beta * (w1 @ xi - v1)) * beta
        g[d + 1] += c * s1
        g[d + 2] += -c
    return g / len(xs)
def grad_F_samples(theta, N=250000):
    dim = len(theta)
    w1 = theta[:d]; v1 = theta[d]; w2 = theta[d + 1]; v2 = theta[d + 2]
    E = rng.normal(size=(N, dim))
    fs = np.empty(N)
    for i in range(N):
        e = E[i]
        w1p = w1 + sigma * e[:d]; v1p = v1 + sigma * e[d]
        w2p = w2 + sigma * e[d + 1]; v2p = v2 + sigma * e[d + 2]
        acc = 0.0
        for xi, ti in zip(xs, ts3):
            s1 = float(w1p @ xi > v1p)
            y = float(w2p * s1 > v2p)
            acc += (1 - 2 * ti) * y + ti ** 2
        fs[i] = acc / len(xs)
    return ((fs - fs.mean())[:, None] * E).mean(axis=0) / sigma
for d1 in [0.5, 1.0, 2.0]:
    v1 = w1_base @ x - d1
    theta = np.concatenate([w1_base, [v1, w2_0, v2_0]])
    gL = grad_L_samples(w1_base, v1, w2_0, v2_0)
    gF = grad_F_samples(theta)
    cos = gL @ gF / np.linalg.norm(gL) / np.linalg.norm(gF)
    print(f"d1={d1}: cos = {cos:.4f}")
