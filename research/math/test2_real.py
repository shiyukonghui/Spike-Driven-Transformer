# -*- coding: utf-8 -*-
"""测试 2: 真实数据小批量等价性 — MNIST + 真实 pe_proj0 权重 + torch autograd
网络: 第一 SPS 级 (conv3x3 s1 p1, 3->32) -> spike(thr) -> 固定随机线性探针 -> CE
      = P1 精确等价类 (单 spike 层 + 线性读出)
SGD 侧: 松弛景观 L~ = CE(probe(sigmoid(4*(conv-thr)))) autograd 解析梯度
ES 侧:  fitness = 硬 CE, MC over theta ~ N(0, sigma^2 I), g = mean((f-fbar)*eps)/sigma
测量: cos(g_sgd, g_es) over sigma 扫描 (R22 分辨率曲线的真实版)
"""
import numpy as np
import torch
import torch.nn.functional as tF

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---- 数据: MNIST npz (cifar10 格式) ----
z = np.load(r'artifacts\mnist\cifar10_data.npz')
print("npz keys:", list(z.keys()))
xt = z['train_x']
yt = z['train_y']
print("x_train shape:", xt.shape, xt.dtype, "max:", float(xt.max()))
B = 128
idx = rng.choice(xt.shape[0], B, replace=False)
if xt.ndim == 4:   # [N,3,32,32] or [N,32,32,3]?
    if xt.shape[1] == 3:
        xb = xt[idx]                      # [B,3,32,32]
    else:
        xb = xt[idx].transpose(0, 3, 1, 2)
else:
    raise SystemExit("unexpected")
yb = yt[idx]
x = torch.tensor(xb, dtype=torch.float32, device=device)
label = torch.tensor(yb, dtype=torch.long, device=device)

# ---- 真实权重: pe_proj0 ----
zw = np.load(r'artifacts\train_burn_init.npz')
w_np = zw['pe_proj0_w'].astype(np.float32)   # [32,3,3,3]
b_np = zw['pe_proj0_b'].astype(np.float32)   # [32]
w = torch.tensor(w_np, requires_grad=True, device=device)
b = torch.tensor(b_np, requires_grad=True, device=device)
print(f"params: w{tuple(w.shape)} b{tuple(b.shape)}  w_std={w.std():.4f}")

thr = 1.0
alpha = 4.0
FDIM = 32 * 32 * 32
probe = (torch.randn(FDIM, 10) * 0.02).to(device)   # 固定随机线性头 (SGD/ES 共用同一目标)

def features_hard(wp, bp):
    h = tF.conv2d(x, wp, bp, stride=1, padding=1)
    return (h > thr).float()

def loss_hard(wp, bp):
    with torch.no_grad():
        s = features_hard(wp, bp).flatten(1)           # [B, 32*32*32]
        logits = s @ probe
        return tF.cross_entropy(logits, label).item(), s.mean().item()

l0, spike_rate = loss_hard(w, b)
print(f"硬损失初值 = {l0:.4f}  spike 率 = {spike_rate:.4f}")

# ---- SGD 侧: 松弛景观解析梯度 (sigmoid 代理) ----
def grad_sgd():
    if w.grad is not None: w.grad = None
    if b.grad is not None: b.grad = None
    h = tF.conv2d(x, w, b, stride=1, padding=1)
    s = torch.sigmoid(alpha * (h - thr))
    logits = s.flatten(1) @ probe
    loss = tF.cross_entropy(logits, label)
    loss.backward()
    return torch.cat([w.grad.flatten(), b.grad.flatten()])

# ---- ES 侧: 硬景观 MC (反对称 + CUDA 大分块) ----
dim = w.numel() + b.numel()
def grad_es(sigma, N=126_000_000):
    """反对称 MC 流式累积: 每块现场生成噪声, g 以 einsum 累加 (从不存全 E)"""
    M = N // 2
    g_acc = torch.zeros(dim, device=device)
    var_acc = torch.zeros(1, device=device)
    CH = 512
    done = 0
    with torch.no_grad():
        while done < M:
            cc = min(CH, M - done)
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
                s = (h > thr).float().reshape(cc * B, FDIM)
                logits = (s @ probe).reshape(cc, B, 10)
                ce = tF.cross_entropy(logits.reshape(cc * B, 10),
                                      label[None, :].expand(cc, B).reshape(cc * B), reduction='none')
                acc += sgn * ce.reshape(cc, B).mean(1)
            g_acc += torch.einsum('c,cd->d', acc, Ec)
            var_acc += (acc ** 2).sum()
            done += cc
    g = g_acc / (N * sigma)
    return g, float(var_acc.sqrt() / np.sqrt(N))

g_sgd = grad_sgd()
print(f"|g_sgd| = {g_sgd.norm():.6f}")
print("=== cos(g_sgd, g_es) vs sigma (真实数据, 反对称 N=126M) ===")
for sigma in [0.05, 0.1]:
    g_es, std_f = grad_es(sigma)
    cos = (g_sgd @ g_es / (g_sgd.norm() * g_es.norm() + 1e-12)).item()
    print(f"sigma={sigma:.3f}: cos = {cos:.4f}  |g_es| = {g_es.norm():.5f}  (硬损失差分 std = {std_f:.4f})", flush=True)



