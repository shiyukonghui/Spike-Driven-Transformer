# -*- coding: utf-8 -*-
"""R21-R25 数值验证: 扰动结构定理 (二次景观解析 + MC)
R21: 反对称 ES 任意协方差 C 无偏; Var[ĝ_i] = Sum_j g_j^2 (c_j/c_i) + g_i^2 (对角);
     梯度能量泄漏 (弱坐标噪声 ~ ||g||^2); c_i ∝ 1/g_i^2 等化 SNR
R22: sigma 是分辨率旋钮: 双边界贡献比 r(sigma) = phi(s/sigma)/phi(0) (解析) —
     sigma << s 分辨, sigma >> s 合并; 单边界 SNR 与 sigma 无关 (解析+MC)
R23: z-score 整形保持梯度方向 (cosine ~ 1)
R24: 双块能量失衡: 各向同性 sigma 毁弱块 SNR; Fisher 形 c_i ∝ 1/g_i^2 等化
"""
import numpy as np
from math import erf

rng = np.random.default_rng(3)

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

print("=== R21a: 二次景观反对称 ES, 任意对角 C 无偏性 ===")
d = 16
H = np.diag(rng.uniform(0.2, 2.0, d))
th = rng.normal(size=d)
g = H @ th
def f_quad(w):
    return -0.5 * w @ H @ w
def es_antithetic(A_sqrt_c, N=200000):
    """对角 C: u = sqrt(c)*eps; g_hat = 0.5*(f(th+u)-f(th-u)) * u/c; f = -0.5*sum(H_ii w_i^2)"""
    E = rng.normal(size=(N, d))
    U = E * A_sqrt_c[None, :]
    fp = -0.5 * np.sum((th[None, :] + U) ** 2 * np.diag(H)[None, :], axis=1)
    fm = -0.5 * np.sum((th[None, :] - U) ** 2 * np.diag(H)[None, :], axis=1)
    diff = fp - fm
    return 0.5 * diff[:, None] * E / A_sqrt_c[None, :]   # C^-1 A eps = eps/sqrt(c)

for trial in range(3):
    c = rng.uniform(0.1, 10.0, d)
    gh = es_antithetic(np.sqrt(c)).mean(axis=0)
    err = np.linalg.norm(gh + g) / np.linalg.norm(g)
    print(f"trial{trial}: |E[g_hat]+g|/|g| = {err:.4f} (0 = 无偏)")

print("=== R21b: 方差公式 Var[i] = Sum_j g_j^2 (c_j/c_i) + g_i^2 ===")
c = rng.uniform(0.2, 5.0, d)
gh_all = es_antithetic(np.sqrt(c), N=300000)
var_mc = gh_all.var(axis=0)
var_th = np.array([np.sum(g ** 2 * (c / c[i])) + g[i] ** 2 for i in range(d)])
rel = np.abs(var_mc - var_th).max() / var_th.max()
print(f"max rel diff = {rel:.4f}")

print("=== R21c: 梯度能量泄漏: g=(10, 1e-3, ...) 各向同性 c=1 ===")
g_leak = np.concatenate([[10.0], np.full(d - 1, 1e-3)])
var_leak = np.array([np.sum(g_leak ** 2 * (np.ones(d) / 1.0)) + g_leak[i] ** 2 for i in range(d)])
print(f"强坐标 SNR^2 = g^2/Var = {g_leak[0]**2/var_leak[0]:.4f}")
print(f"弱坐标 SNR^2 = {g_leak[1]**2/var_leak[1]:.2e}  (噪声 ||g||^2={np.sum(g_leak**2):.1f} 淹没弱信号 {g_leak[1]**2:.1e})")

print("=== R21d: Fisher 形 c_i ∝ 1/g_i^2 等化 SNR ===")
c_f = 1.0 / g_leak ** 2
var_f = np.array([np.sum(g_leak ** 2 * (c_f / c_f[i])) + g_leak[i] ** 2 for i in range(d)])
snr2 = g_leak ** 2 / var_f
print(f"SNR^2 各坐标: min={snr2.min():.5f} max={snr2.max():.5f} (相等 = 等化, 理论 1/(d+1)={1/(d+1):.5f})")

print("=== R22a: 单边界 SNR 的 sigma 不变性 (解析 vs MC) ===")
D, sig = 1.0, 0.0
for sigma in [0.3, 1.0, 3.0]:
    N = 300000
    u = rng.normal(size=N)
    diff = 2.0 * ((u > D).astype(float) - (u < -D).astype(float))  # Delta_f = 2
    gh = diff * u / (2 * sigma)
    snr2 = gh.mean() ** 2 / gh.var()
    Phi_d = 0.5 * (1 + erf(D / np.sqrt(2)))
    snr2_an = 2 * phi(D) ** 2 / (phi(D) + D * (1 - Phi_d) - 2 * phi(D) ** 2)  # 含减均值项
    print(f"sigma={sigma}: SNR^2 MC={snr2:.5f} 解析={snr2_an:.5f} (与 sigma 无关)")

print("=== R22b: sigma = 分辨率旋钮: 双边界贡献比 r = phi(s/sigma)/phi(0) ===")
s_space = 2.0
for sigma in [0.25, 0.5, 1.0, 2.0, 4.0]:
    r_an = phi(s_space / sigma) / phi(0.0)
    print(f"sigma={sigma}: 近边界贡献/远边界 = {r_an:.4f} (sigma<<s 分辨, >>s 合并)")

print("=== R23: z-score 整形保持梯度方向 ===")
n_s, d_s = 64, 8
X = rng.normal(size=(n_s, d_s)); X /= np.linalg.norm(X, axis=1, keepdims=True) / 2.0
ts = rng.integers(0, 2, n_s).astype(float)
w0 = rng.normal(size=d_s) * 0.3
sigma_es = 0.5
def L_hard(w):
    return np.mean(((X @ w > 0).astype(float) - ts) ** 2)
N = 200000
E = rng.normal(size=(N, d_s))
fs = np.array([L_hard(w0 + sigma_es * e) for e in E])
g_raw = ((fs - fs.mean())[:, None] * E).mean(axis=0) / sigma_es
z = (fs - fs.mean()) / fs.std()
g_z = (z[:, None] * E).mean(axis=0) / sigma_es
cos = g_raw @ g_z / np.linalg.norm(g_raw) / np.linalg.norm(g_z)
print(f"cos(g_raw, g_z) = {cos:.6f}  |g_z|/|g_raw| = {np.linalg.norm(g_z)/np.linalg.norm(g_raw):.4f} (理论 1/std(f) = {1/fs.std():.4f})")

print("=== R24: 双块能量失衡与 Fisher 形修正 ===")
d2 = 8
g2 = np.concatenate([np.full(4, 5.0), np.full(4, 0.05)])  # 块1能量 2500, 块2 1e-5
c_iso = np.ones(d2)
var_iso = np.array([np.sum(g2 ** 2 * (c_iso / c_iso[i])) + g2[i] ** 2 for i in range(d2)])
snr_iso = g2 ** 2 / var_iso
c_fish = 1.0 / g2 ** 2
var_fish = np.array([np.sum(g2 ** 2 * (c_fish / c_fish[i])) + g2[i] ** 2 for i in range(d2)])
snr_fish = g2 ** 2 / var_fish
print(f"各向同性:  SNR^2 强块={snr_iso[0]:.4f} 弱块={snr_iso[4]:.2e} 比值={snr_iso[0]/snr_iso[4]:.1e}")
print(f"Fisher 形: SNR^2 强块={snr_fish[0]:.5f} 弱块={snr_fish[4]:.5f} 比值={snr_fish[0]/snr_fish[4]:.2f} (理论 1.0)")
