# -*- coding: utf-8 -*-
"""R26-R30 数值验证: 突破点综合
R26: 自适应阈值 vs 固定阈值 的活跃分数随膜分布展宽的标度
     固定: E_m[p_live] = phi(0)/sqrt(w^2+s^2+sigma_m^2) (Gauss-Gauss 卷积)
     自适应: phi(0)/sqrt(w^2+s^2) — 对 sigma_m 平坦 (漂移免疫)
R28: 边界间距匹配: p_flip(sigma) = Phi(-d/(sigma*||a||)) — 翻转敏感尺度 sigma* = d/||a||
     (参数空间边界间距) — 验证 MC vs 解析; 与代理宽度 sigma_b=1/(beta*||a||) 对偶
"""
import numpy as np
from math import erf

rng = np.random.default_rng(4)

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

def Phi(u):
    return 0.5 * (1 + np.vectorize(erf)(u / np.sqrt(2)))

print("=== R26: 活跃分数的膜展宽标度 (w=0.1, s=0.3) ===")
w, s2 = 0.1, 0.09
adaptive = phi(0.0) / np.sqrt(w**2 + s2)
print(f"{'sigma_m':>8} {'固定阈值 E[p_live]':>20} {'解析 1/sqrt(w^2+s^2+sm^2)':>26} {'自适应':>10}")
for sm in [0.5, 1.0, 2.0, 4.0]:
    ms = rng.normal(0, sm, 400000)
    mc = np.mean(phi(ms / np.sqrt(w**2 + s2)) / np.sqrt(w**2 + s2))
    an = phi(0.0) / np.sqrt(w**2 + s2 + sm**2)
    print(f"{sm:>8.1f} {mc:>20.5f} {an:>26.5f} {adaptive:>10.5f}")
print(f"自适应阈值对 sigma_m 平坦 (漂移免疫); 固定阈值按 1/sqrt(w^2+s^2+sm^2) 衰减")
print(f"sigma_m=4 时自适应/固定增益 = {adaptive/(phi(0.0)/np.sqrt(w**2+s2+16)):.1f}x")

print("=== R28: 翻转敏感尺度 = 参数空间边界间距 sigma* = d/||a|| ===")
d_par = 12
a = rng.normal(size=d_par) * (X := rng.choice([0.0, 1.0], d_par, p=[0.7, 0.3]))  # 输入 spike 向量
na = np.linalg.norm(a)
th0 = rng.normal(size=d_par) * 0.1
d_mem = 0.8  # 当前膜距 (h - v > 0 侧)
# 调 th0 使 th0.a - v = d_mem
v = th0 @ a - d_mem
print(f"||a||={na:.3f}  理论 sigma*=d/||a||={d_mem/na:.4f}")
for sigma in [0.05, 0.1, d_mem / na, 0.4, 0.8]:
    N = 300000
    E = rng.normal(size=(N, d_par))
    h_new = (th0[None, :] + sigma * E) @ a - v
    p_mc = (h_new < 0).mean()
    p_an = Phi(-d_mem / (sigma * na))
    print(f"sigma={sigma:.4f}: p_flip MC={p_mc:.5f} 解析 Phi(-d/(s*||a||))={p_an:.5f}")
print("翻转概率在 sigma ~ d/||a|| 处过渡 — 每边界的参数空间间距由 ||a||=||dh/dθ|| 决定;")
print("代理宽度 sigma_b = 1/(beta*||a||) 与 ES 全局 sigma 的匹配 = Fisher 形协方差 (R24/R25)")

print("=== R26b: 深度复合下的自适应阈值收益 (L=16, 漂移 |N(0,1)|) ===")
L = 16
deltas = np.abs(rng.normal(0, 1.0, (4000, L)))
pl_fixed = (phi(deltas / np.sqrt(w**2 + s2)) / np.sqrt(w**2 + s2)).prod(axis=1)
pl_adapt = (phi(0.0) / np.sqrt(w**2 + s2)) ** L
print(f"固定阈值中位信号={np.median(pl_fixed):.3e}  自适应={pl_adapt:.3e}  中位增益={np.median(pl_adapt/np.maximum(pl_fixed,1e-300)):.1f}x")
