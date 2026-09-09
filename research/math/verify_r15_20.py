# -*- coding: utf-8 -*-
"""R15-R20 数值验证: 膜统计死区定理 + 时间维度通道
R15: 活跃分数卷积公式 p_live = phi((m-v)/sqrt(w^2+s^2))/sqrt(w^2+s^2)  (MC 验证)
R16: 自适应阈值 vs 固定最优阈值 的增益
R17: 膜漂移的指数衰减表
R18: 深度复合: L 层漂移膜的乘性塌缩
R19: 静态编码 vs Poisson 编码的独立决策数 (近阈值访问)
R20: Poisson 计数的 Binomial Fisher ∝ T; 静态退化为单标量通道
"""
import numpy as np

rng = np.random.default_rng(2)

def phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

print("=== R15: 活跃分数卷积公式 (Gauss-Gauss 卷积) ===")
w, s2 = 0.1, 1.0
se = np.sqrt(w**2 + s2)
for (m, v) in [(0.0, 0.0), (0.5, 0.0), (1.5, 0.0)]:
    h = rng.normal(m, np.sqrt(s2), size=400000)
    mc = np.mean(phi((h - v) / w) / w)
    an = phi((m - v) / se) / se
    print(f"m={m}: MC={mc:.5f} 解析={an:.5f} 比值={mc/an:.4f}")

print("=== R16: 自适应阈值增益 (m_i ~ N(0,1), M=256 神经元) ===")
M = 256
ms = rng.normal(0, 1, M)
s2n, wn = 1.0, 0.1
sen = np.sqrt(wn**2 + s2n)
def avg_live(vfix):
    return np.mean(phi((ms - vfix) / sen) / sen)
vs = np.linspace(-2, 2, 401)
best = max(avg_live(v) for v in vs)
adapt = np.mean(phi(0.0) / sen)   # v_i = m_i => 距离恒 0
print(f"固定最优阈值 avg_live={best:.5f}  自适应 v_i=m_i avg_live={adapt:.5f}  增益={adapt/best:.2f}x")

print("=== R17: 膜漂移指数衰减 (w=0.1, s=1) ===")
for dd in [0.0, 0.5, 1.0, 2.0, 3.0]:
    pl = phi(dd / se) / se
    print(f"漂移 m-v={dd}: p_live={pl:.5f} 相对增益={pl/(phi(0)/se):.4f} (预测 exp(-d^2/2s^2)={np.exp(-dd*dd/2):.4f})")

print("=== R18: 深度复合: L 层漂移乘性塌缩 ===")
for L in [1, 4, 8, 16]:
    deltas = np.abs(rng.normal(0, 1.0, (2000, L)))
    pl = phi(deltas / se) / se
    sig = pl.prod(axis=1)
    cent = (phi(0.0) / se) ** L
    print(f"L={L}: 漂移网络中位信号={np.median(sig):.3e} 居中网络={cent:.3e} 中位增益={np.median(cent/np.maximum(sig,1e-300)):.1f}x")

print("=== R19: 时间通道容量: 可达 spike 模式数 (静态 O(logT) vs Poisson O(T)) ===")
T, gamma, vt = 8, 0.7, 1.0
# 静态: s = 1[t >= tau] (单调轨迹, 由穿越时刻 tau 唯一决定) => 可达模式 = T+1
static_patterns = set()
for I in np.linspace(0.01, 3.0, 3000):
    h, pat = 0.0, []
    for t in range(T):
        h = gamma * h + I
        pat.append(int(h > vt))
    static_patterns.add(tuple(pat))
# Poisson: 每步独立噪声 => 采样可达模式数
pois_patterns = set()
for _ in range(20000):
    h, pat = 0.0, []
    for t in range(T):
        h = gamma * h + 0.35 + rng.normal(0, 0.6)
        pat.append(int(h > vt))
    pois_patterns.add(tuple(pat))
print(f"T={T}: 静态可达模式 = {len(static_patterns)} (理论 T+1={T+1}, 容量 log2={np.log2(T+1):.1f} bit)")
print(f"      Poisson 可达模式 = {len(pois_patterns)} (上限 2^T={2**T}, 容量 ~T bit)")

print("=== R20: 计数统计量的内在方差: Poisson ~ T·p(1-p) 过散; 静态 = 0 (确定性) ===")
T2 = 32
for noise_std in [0.6, 0.0]:
    trials = 4000
    h = np.zeros(trials); c = np.zeros(trials)
    for t in range(T2):
        h = gamma * h + 0.35 + (rng.normal(0, noise_std, trials) if noise_std > 0 else 0.0)
        c += (h > vt).astype(float)
    p = c.mean() / T2
    print(f"noise={noise_std}: Var[C]={c.var():.2f}  T·p(1-p)={T2*p*(1-p):.1f}  过散比={c.var()/max(T2*p*(1-p),1e-9):.2f}")
print("解读: 静态编码下 C 是 (w,x) 的确定性函数 (Var=0) — T 步是同一标量的 T 次确定性读取;")
print("      Poisson 每步注入独立噪声 — T 个部分独立的 Bernoulli 决策, 信息容量 ∝ T")
