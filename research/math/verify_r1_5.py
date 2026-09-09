# -*- coding: utf-8 -*-
"""R1-R5 数值验证 v2（修正: 损失因子 (1-2t); 纯核验证放 spike-概率景观）
单神经元: y=1[w.x>v] (真) / y~=sigmoid(beta*(w.x-v)) (松弛)
spike-概率景观: E[y](w) — 无损失权重, 纯边界核
  ES 平滑:   F_sigma(w) = Phi((w.x-v)/(sigma*||x||)),  grad = phi(z)*x/(sigma*||x||)
  松弛网络:  E~[y~](w) = sigmoid(beta*(w.x-v)),        grad = beta*phi(beta*(w.x-v))*x
  宽度: w_ES = sigma*||x|| (w.x 单位),  w_relax = 1/beta — 相等 <=> sigma*||x||*beta = 1
损失景观: l=(y-t)^2, y in{0,1} => l=(1-2t)y+t^2, grad_E[l] = (1-2t)*grad_E[y]
"""
import numpy as np

rng = np.random.default_rng(0)
d = 8
x = rng.normal(size=d)
nx = np.linalg.norm(x)
t = 1.0
v = 0.0

def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -30, 30)))

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

def L_true(w):
    return (float(w @ x > v) - t) ** 2

def es_grad(w, sigma, f, N=400000):
    eps = rng.normal(size=(N, d))
    fs = np.array([f(w + sigma * e) for e in eps])
    return ((fs - fs.mean())[:, None] * eps).mean(axis=0) / sigma

def fd_grad(w, f, h=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g

print("=== (a) 真景观 ES -> 解析 grad Phi((w.x-v)/(sigma*||x||)), 含损失因子 (1-2t) ===")
w = np.zeros(d)
for sigma in [0.3, 1.0, 3.0]:
    z = (w @ x - v) / (sigma * nx)
    g_an = (1 - 2 * t) * phi(z) * x / (sigma * nx)
    g_mc = es_grad(w, sigma, L_true)
    rel = np.linalg.norm(g_mc - g_an) / np.linalg.norm(g_an)
    print(f"sigma={sigma}: |g_mc|={np.linalg.norm(g_mc):.5f} |g_an|={np.linalg.norm(g_an):.5f} rel_err={rel:.4f}")

print("=== (b) 松弛 spike-概率景观: FD vs 解析 (纯核, logistic) ===")
beta = 6.0
def grad_Ey_relax(w):
    s = sigmoid(beta * (w @ x - v))
    return s * (1 - s) * beta * x
g_fd = fd_grad(w, lambda u: sigmoid(beta * (u @ x - v)))
g_an = grad_Ey_relax(w)
print("max abs diff:", np.abs(g_fd - g_an).max())

print("=== (c) probit 代理: 两核逐位恒等 (Gauss=Gauss) ===")
from math import erf
def Phi(u):  # 标准正态 CDF
    return 0.5 * (1 + np.vectorize(erf)(u / np.sqrt(2)))
def grad_Ey_probit(w):
    return beta * phi(beta * (w @ x - v)) * x
g_fd = fd_grad(w, lambda u: float(Phi(beta * (u @ x - v))))
g_an = grad_Ey_probit(w)
print("max abs diff:", np.abs(g_fd - g_an).max())

print("=== (c2) 宽度对齐 sigma*||x|| = 1/beta: 真景观核 vs probit 松弛核 ===")
sigma_c = 1.0 / (beta * nx)
g_true = phi(0.0) * x / (sigma_c * nx)          # 真景观平滑核 (膜距 0)
g_probit = grad_Ey_probit(w)                     # probit 松弛核 (膜距 0)
print(f"|g_true|={np.linalg.norm(g_true):.5f} |g_probit|={np.linalg.norm(g_probit):.5f} max diff={np.abs(g_true-g_probit).max():.2e}")
print(f"(sigmoid 峰值常数 0.25 vs Gauss 0.3989: 比值={0.25/phi(0.0):.4f})")

print("=== (d) 纯核膜距扫描 (logistic): |grad| = beta*s(1-s)*||x|| ===")
for m in [0.0, 0.5 / beta, 1.0 / beta, 2.0 / beta]:  # m = 膜距 (w.x 单位)
    w2 = w + (m / nx) * x / nx  # 沿法向推到膜距 m
    g = grad_Ey_relax(w2)
    s = sigmoid(beta * m)
    K = beta * s * (1 - s) * nx
    print(f"膜距(w.x 单位)={m:.4f}: |grad|={np.linalg.norm(g):.5f} 预测={K:.5f} 比值={np.linalg.norm(g)/K:.4f}")
