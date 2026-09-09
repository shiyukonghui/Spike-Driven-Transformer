import numpy as np
import torch
import torch.nn.functional as tF
torch.manual_seed(0)
rng = np.random.default_rng(0)
device = 'cuda'
z = np.load(r'artifacts\mnist\cifar10_data.npz')
xt, yt = z['train_x'], z['train_y']
B = 8
idx = rng.choice(xt.shape[0], B, replace=False)
x = torch.tensor(xt[idx], dtype=torch.float32, device=device)
label = torch.tensor(yt[idx], dtype=torch.long, device=device)
zw = np.load(r'artifacts\train_burn_init.npz')
w = torch.tensor(zw['pe_proj0_w'].astype(np.float32), device=device)
b = torch.tensor(zw['pe_proj0_b'].astype(np.float32), device=device)
thr, alpha = 1.0, 4.0
FDIM = 32*32*32
probe32 = (torch.randn(FDIM, 10) * 0.02).to(device)
probe = probe32.half()
patches = tF.unfold(x, kernel_size=3, padding=1).transpose(1, 2).reshape(B*1024, 27)
pT_base = patches.t().contiguous().half()             # [27, B*1024] fp16
def loss_from_out(o):
    s = o.permute(0, 2, 1).reshape(-1, FDIM)          # [C*B, F] fp16
    logits = (s @ probe).float().reshape(-1, B, 10)
    ce = tF.cross_entropy(logits.reshape(-1, 10), label.repeat(logits.shape[0]), reduction='none')
    return ce.reshape(-1, B).mean(1)
def grad_sgd():
    wq = w.clone().requires_grad_(True); bq = b.clone().requires_grad_(True)
    h = tF.conv2d(x, wq, bq, stride=1, padding=1)
    s = torch.sigmoid(alpha * (h - thr))
    logits = s.flatten(1) @ probe32
    tF.cross_entropy(logits, label).backward()
    return torch.cat([wq.grad.flatten(), bq.grad.flatten()])
g_sgd = grad_sgd()
print(f"|g_sgd| = {g_sgd.norm():.6f}", flush=True)
def grad_es_full(sigma, N=8_000_000):
    M = N // 2
    g_acc = torch.zeros(w.numel()+b.numel(), device=device)
    CH = 1024
    done = 0
    with torch.no_grad():
        while done < M:
            cc = min(CH, M - done)
            Ec = torch.randn(cc, w.numel()+b.numel(), device=device)
            acc = torch.zeros(cc, device=device)
            for sgn in (+1.0, -1.0):
                Pw = (w[None] + sgn*sigma*Ec[:, :w.numel()].reshape(-1, *w.shape)).half()
                Pb = (b[None] + sgn*sigma*Ec[:, w.numel():]).half()
                Wmat = Pw.reshape(cc, 32, 27)
                h = torch.matmul(Wmat, pT_base).float() + Pb[:, :, None].float()
                o = (h > thr).half()
                acc += sgn * loss_from_out(o)
            g_acc += torch.einsum('c,cd->d', acc.float(), Ec)
            done += cc
    return g_acc / (N * sigma)
def grad_es_bias(sigma, N=8_000_000):
    """bias-only: conv ?????, ????? elementwise"""
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
                hb = h0[None] + sgn*sigma*Ec[:, :, None, None, None]
                o = (hb > thr).half().reshape(cc*B, FDIM)
                logits = (o @ probe).float().reshape(cc, B, 10)
                ce = tF.cross_entropy(logits.reshape(-1, 10), label.repeat(cc), reduction='none')
                acc += sgn * ce.reshape(cc, B).mean(1)
            g_acc += torch.einsum('c,cd->d', acc, Ec)
            done += cc
    return g_acc / (N * sigma)
g1 = grad_es_full(0.05); g2 = grad_es_full(0.05); g3 = grad_es_full(0.1)
def cos(a, b): return (a @ b / (a.norm()*b.norm() + 1e-12)).item()
r = cos(g1, g2)
print(f"full 928d: sigma=0.05 raw cos = {cos(g_sgd,g1):.4f}  reliability = {r:.4f}  disatt = {cos(g_sgd,g1)/np.sqrt(max(r,1e-9)):.4f}")
print(f"full 928d: sigma=0.10 raw cos = {cos(g_sgd,g3):.4f}", flush=True)
# bias-only ???
gs_b = g_sgd[w.numel():]
eb1 = grad_es_bias(0.05); eb2 = grad_es_bias(0.05)
rb = cos(eb1, eb2)
print(f"bias 32d:  raw cos = {cos(gs_b,eb1):.4f}  reliability = {rb:.4f}  disatt = {cos(gs_b,eb1)/np.sqrt(max(rb,1e-9)):.4f}")
print(f"bias 32d:  |e1|={eb1.norm():.4f} |gs_b|={gs_b.norm():.4f}")
