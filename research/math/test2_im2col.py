import numpy as np
import torch
import torch.nn.functional as tF
torch.manual_seed(0)
rng = np.random.default_rng(0)
device = 'cuda'
z = np.load(r'artifacts\mnist\cifar10_data.npz')
xt, yt = z['train_x'], z['train_y']
B = 128
idx = rng.choice(xt.shape[0], B, replace=False)
xb = xt[idx]; yb = yt[idx]
x = torch.tensor(xb, dtype=torch.float32, device=device)
label = torch.tensor(yb, dtype=torch.long, device=device)
zw = np.load(r'artifacts\train_burn_init.npz')
w = torch.tensor(zw['pe_proj0_w'].astype(np.float32), requires_grad=True, device=device)
b = torch.tensor(zw['pe_proj0_b'].astype(np.float32), requires_grad=True, device=device)
thr, alpha = 1.0, 4.0
FDIM = 32*32*32
probe = (torch.randn(FDIM, 10) * 0.02).to(device)
# im2col ??: patches [B*1024, 27], 连续化转置供广播 matmul (零拷贝)
patches = tF.unfold(x, kernel_size=3, padding=1)      # [B, 27, 1024]
patches = patches.transpose(1, 2).reshape(B*1024, 27) # [B*1024, 27]
pT_base = patches.t().contiguous()                     # [27, B*1024]
def out_hard(wp, bp, chunk_dim):
    """wp [C,32,3,3,3] -> o [C, 32, B*1024] (hard spike)"""
    C = wp.shape[0]
    Wmat = wp.reshape(C, 32, 27)                       # [C, 32, 27]
    h = torch.matmul(Wmat, pT_base) + bp[:, :, None]   # [C, 32, B*1024]
    return (h > thr).float()
def loss_from_out(o):
    s = o.permute(0, 2, 1).reshape(-1, FDIM)           # [C*B, F]  (f = ch*1024 + pos)
    logits = (s @ probe).reshape(-1, B, 10)
    ce = tF.cross_entropy(logits.reshape(-1, 10), label.repeat(logits.shape[0]), reduction='none')
    return ce.reshape(-1, B).mean(1)                   # [C]
def grad_sgd():
    if w.grad is not None: w.grad = None
    if b.grad is not None: b.grad = None
    h = tF.conv2d(x, w, b, stride=1, padding=1)
    s = torch.sigmoid(alpha * (h - thr))
    logits = s.flatten(1) @ probe
    tF.cross_entropy(logits, label).backward()
    return torch.cat([w.grad.flatten(), b.grad.flatten()])
g_sgd = grad_sgd()
print(f"|g_sgd| = {g_sgd.norm():.6f}", flush=True)
def grad_es_bias(sigma, N=2_000_000):
    """bias-only 高精度对照: conv 预计算一次, 候选评估纯 elementwise (fp32)"""
    with torch.no_grad():
        h0 = tF.conv2d(x, w, b, stride=1, padding=1)   # [B,32,32,32]
    M = N // 2
    g_acc = torch.zeros(32, device=device)
    CH = 4096
    done = 0
    with torch.no_grad():
        while done < M:
            cc = min(CH, M - done)
            Ec = torch.randn(cc, 32, device=device)
            acc = torch.zeros(cc, device=device)
            for sgn in (+1.0, -1.0):
                hb = h0[None] + sgn * sigma * Ec[:, None, :, None, None, None]  # [cc,B,32,32,32]
                o = (hb > thr).float().reshape(cc * B, FDIM)
                logits = (o @ probe32).reshape(cc, B, 10)
                ce = tF.cross_entropy(logits.reshape(-1, 10), label.repeat(cc), reduction='none')
                acc += sgn * ce.reshape(cc, B).mean(1)
            g_acc += torch.einsum('c,cd->d', acc, Ec)
            done += cc
    return g_acc / (N * sigma)
dim = w.numel() + b.numel()
def grad_es(sigma, N=8_000_000):
    M = N // 2
    g_acc = torch.zeros(dim, device=device)
    var_acc = 0.0
    CH = 256
    done = 0
    with torch.no_grad():
        while done < M:
            cc = min(CH, M - done)
            Ec = torch.randn(cc, dim, device=device)
            acc = torch.zeros(cc, device=device)
            for sgn in (+1.0, -1.0):
                Pw = w[None] + sgn * sigma * Ec[:, :w.numel()].reshape(-1, *w.shape)
                Pb = b[None] + sgn * sigma * Ec[:, w.numel():]
                o = out_hard(Pw, Pb, cc)
                acc += sgn * loss_from_out(o)
            g_acc += torch.einsum('c,cd->d', acc, Ec)
            var_acc += float((acc**2).sum())
            done += cc
    return g_acc / (N * sigma), np.sqrt(var_acc / N)
g1, s1 = grad_es(0.05)
g2, _ = grad_es(0.05)
g3, _ = grad_es(0.1)
def cos(a, b): return (a @ b / (a.norm()*b.norm() + 1e-12)).item()
r = cos(g1, g2)
raw = cos(g_sgd, g1)
raw3 = cos(g_sgd, g3)
print(f"sigma=0.05: raw cos = {raw:.4f}  reliability = {r:.4f}  disattenuated = {raw/np.sqrt(max(r,1e-9)):.4f}", flush=True)
print(f"sigma=0.10: raw cos = {raw3:.4f}", flush=True)
print(f"|g1|={g1.norm():.4f} |g2|={g2.norm():.4f} |g3|={g3.norm():.4f} |g_sgd|={g_sgd.norm():.4f}", flush=True)
# bias-only 高精度对照 (fp32)
gs_b = g_sgd[w.numel():]
eb1 = grad_es_bias(0.05)
eb2 = grad_es_bias(0.05)
rb = cos(eb1, eb2)
print(f"bias-only 32d: raw cos = {cos(gs_b,eb1):.4f}  reliability = {rb:.4f}  disatt = {cos(gs_b,eb1)/np.sqrt(max(rb,1e-9)):.4f}")
print(f"bias-only 32d: |e1|={eb1.norm():.4f} |gs_b|={gs_b.norm():.4f}")

