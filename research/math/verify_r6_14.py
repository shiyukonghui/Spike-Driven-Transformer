# -*- coding: utf-8 -*-
"""R6-R14 数值验证 v2 (修正: 二值网络平滑损失对 spike 概率线性 — E[(y-t)^2]=(1-2t)E[y]+t^2)
R6: 数据集平滑景观 = 每样本边界核之和(权重含 (1-2t_i)/||x_i||), ES MC -> 解析
R7: ES 估计误差各向同性 (参数空间坐标)
R8: 活跃集集中度随 sigma 收缩
R9-R14: L 级联 AND(SGD 乘性) vs OR(ES 并集) 深度标度
"""
import numpy as np
from math import erf

rng = np.random.default_rng(1)

def Phi(u):
    return 0.5 * (1 + np.vectorize(erf)(u / np.sqrt(2)))

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

n, d = 64, 8
X = rng.normal(size=(n, d))
X /= np.linalg.norm(X, axis=1, keepdims=True) / 2.0
nx1 = np.linalg.norm(X, axis=1)
ts = rng.integers(0, 2, size=n).astype(float)
v = 0.0
w0 = rng.normal(size=d) * 0.3
sigma = 0.5

def L_correct(w):
    z = (X @ w - v) / (sigma * nx1)
    return np.mean((1 - 2 * ts) * Phi(z) + ts ** 2)

def grad_exact(w):
    z = (X @ w - v) / (sigma * nx1)
    coef = (1 - 2 * ts) * phi(z) / (sigma * nx1)
    return (coef[:, None] * X).mean(axis=0)

def L_hard(w):
    return np.mean(((X @ w > v).astype(float) - ts) ** 2)

def es_grad(w, N):
    eps = rng.normal(size=(N, d))
    fs = np.array([L_hard(w + sigma * e) for e in eps])
    return ((fs - fs.mean())[:, None] * eps).mean(axis=0) / sigma

print("=== R6: ES MC -> 每样本解析核之和 (修正景观) ===")
g_ex = grad_exact(w0)
g_es = es_grad(w0, 400000)
print(f"|g_exact|={np.linalg.norm(g_ex):.5f} |g_es|={np.linalg.norm(g_es):.5f} rel_err={np.linalg.norm(g_es-g_ex)/np.linalg.norm(g_ex):.4f}")

print("=== R7: ES 误差各向同性: 每坐标平方误差 ===")
errs = np.mean([ (es_grad(w0, 20000) - g_ex)**2 for _ in range(60) ], axis=0)
print(f"per-coord sq-err: min={errs.min():.2e} max={errs.max():.2e} max/min={errs.max()/errs.min():.2f} (~1 = 各向同性)")

print("=== R8: 活跃集集中度: 前10%样本梯度质量 vs sigma ===")
for s_ in [0.1, 0.25, 0.5, 1.0]:
    z = (X @ w0 - v) / (s_ * nx1)
    mass = np.abs((1 - 2 * ts) * phi(z) / (s_ * nx1))
    order = np.argsort(mass)[::-1]
    top10 = mass[order[: n // 10]].sum() / mass.sum()
    print(f"sigma={s_}: top-10% 样本承载 {top10:.3f}")

print("=== R9-R14: L 级联 AND vs OR 深度标度 (风格化模型) ===")
def cascade(L, p, trials=4000):
    sgd = es_ = 0.0
    for _ in range(trials):
        live = rng.random(L) < p
        if live.all():
            sgd += np.prod(rng.uniform(0.5, 1.5, L))
        if live.any():
            es_ += 1.0
    return sgd / trials, es_ / trials

print(f"{'L':>4} {'p=.3 SGD':>10} {'p=.3 ES':>10} {'ES/SGD':>9} | {'p=.7 SGD':>10} {'p=.7 ES':>10} {'ES/SGD':>9}")
for L in [1, 4, 8, 16, 32]:
    s3, e3 = cascade(L, 0.3)
    s7, e7 = cascade(L, 0.7)
    print(f"{L:>4} {s3:>10.5f} {e3:>10.5f} {e3/max(s3,1e-9):>9.1f} | {s7:>10.5f} {e7:>10.5f} {e7/max(s7,1e-9):>9.1f}")
print("理论: SGD ~ p^L (指数衰减), ES ~ 1-(1-p)^L (并集良态); ES/SGD ~ 指数增长")
