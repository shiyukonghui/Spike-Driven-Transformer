# -*- coding: utf-8 -*-
"""
PyTorch 小规模训练对照脚本（对应规格 Task 6）。

目的：与 rust-sdt（Burn 端）训练对照脚本保持完全一致的数据、模型同构形式与指标口径：
  - 数据：直接读取 rust-sdt/artifacts/cifar10_data.npz（与 Burn 端同一份 8000/2000 子集）；
  - 模型：FusedSDT —— 与 Burn 端「BN 已融合、Conv(w,b) + LIF、无可训练 BN」的形式严格同构，
    前向计算图与 export_reference.py 的手动展开链路（SPS -> blocks -> head）逐算子对应；
  - 权重：--init-npz 从 rust-sdt/artifacts/sdt_reference.npz 加载「融合形式」权重
    （pe_proj{i}_w/b、pe_rpe_w/b、blk{j}_q/k/v/proj/fc1/fc2_w/b、th_w、head_w/head_b），
    由于 Conv+BN 已融合为 W'/B'，直接作为 Conv 的 weight/bias 使用，无需拆回 BN；
  - 静态校准（默认开启）：NPZ 中的融合权重基于「初始 BN running stats」（mu=0,var=1），
    与真实激活分布严重不匹配（首层 LIF 输入幅值 ~0.03，远低于阈值 1.0），直接使用会导致
    全网零脉冲死锁（无法训练）。校准对每个喂给 LIF 的卷积/线性层，用训练子集测量其输出
    每通道 (mu, sigma)，做 (W',B') -> (W'/sigma, (B'-mu)/sigma) 的等效替换——数学上严格
    等价于「用真实统计量重新融合 BN」，网络形式保持纯 Conv(bias)+LIF，与 Burn 端同构性
    不变（校准后的初始权重可用 --save-init 导出供 Burn 侧复现）；
  - LIF：spikingjelly MultiStepLIFNode(backend="torch")（支持反向传播，可训练）；
  - 输出：每 epoch 在 2000 张测试图上评估 top-1，写入 CSV（epoch,train_loss,val_top1），
    格式与 Burn 端训练对照完全一致。

运行方式（PowerShell，仓库根目录）:
  $env:PYTHONPATH="f:\RustProjects\Spike-Driven-Transformer\.pylibs;f:\RustProjects\Spike-Driven-Transformer"
  D:\Anaconda\envs\Pytorch-CUDA\python.exe scripts\train_pytorch_reference.py --epochs 2 --init-npz rust-sdt\artifacts\sdt_reference.npz
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 保证项目根目录可导入（.pylibs 内的 timm / spikingjelly）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spikingjelly.clock_driven.neuron import MultiStepLIFNode  # noqa: E402
from spikingjelly.clock_driven import functional  # noqa: E402

# ---------------------------------------------------------------------------
# 常量（与 scripts/export_cifar10.py / rust-sdt 端约定保持一致）
# ---------------------------------------------------------------------------
T_STEPS = 4              # SNN 时间步数
PATCH_SIZE = 16          # token 化下采样倍率
NUM_CLASSES = 10         # CIFAR-10 类别数
DEFAULT_DATA_NPZ = os.path.join("rust-sdt", "artifacts", "cifar10_data.npz")


# ---------------------------------------------------------------------------
# 模型：FusedSDT（与 Burn 端融合形式严格同构）
# ---------------------------------------------------------------------------

class FusedSPS(nn.Module):
    """与 module/sps.py MS_SPS 逐算子同构的融合版 patch embed。

    差异：Conv2d(bias=True) + BN 融合为单个 Conv（权重 W'、偏置 B'），无 BN 层。
    4 级 stage 顺序：conv -> LIF -> [pool]（stage0/1 不池化），
    stage2 为 LIF -> pool，stage3 为 pool -> LIF（与 pooling_stat="0011" 展开一致）。
    最后 rpe 分支：rpe_conv -> (+x_feat 残差)。
    注意：与 PyTorch 原始模型一致，rpe_conv 之后无 LIF（rpe_lif 不在前向路径中）。
    """

    def __init__(self, img_h: int, img_w: int, in_ch: int, embed_dims: int):
        super().__init__()
        e = embed_dims
        # 与 MS_SPS 相同的卷积核/步长/填充配置，但 bias=True 承载融合后的 B'
        self.proj_conv0 = nn.Conv2d(in_ch, e // 8, 3, 1, 1, bias=True)
        self.proj_conv1 = nn.Conv2d(e // 8, e // 4, 3, 1, 1, bias=True)
        self.proj_conv2 = nn.Conv2d(e // 4, e // 2, 3, 1, 1, bias=True)
        self.proj_conv3 = nn.Conv2d(e // 2, e, 3, 1, 1, bias=True)
        self.rpe_conv = nn.Conv2d(e, e, 3, 1, 1, bias=True)
        # 与原模型一致的 LIF（torch 后端，可反向传播；v_threshold 默认 1.0）
        self.lif0 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        # 与 MS_SPS 相同的 3x3/stride2/pad1 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1,
                                    ceil_mode=False)

    def forward(self, x):
        """输入 [T,B,3,32,32] -> 输出 [T,B,embed_dims,H/16,W/16]（token 特征图）。"""
        T, B, _, H, W = x.shape
        ratio = 1
        # 阶段 0：conv -> LIF（pooling_stat[0]=='0'，不池化）
        x = self.proj_conv0(x.flatten(0, 1))
        x = self.lif0(x.reshape(T, B, -1, H // ratio, W // ratio))
        x = x.flatten(0, 1).contiguous()
        # 阶段 1：conv -> LIF（pooling_stat[1]=='0'，不池化）
        x = self.proj_conv1(x)
        x = self.lif1(x.reshape(T, B, -1, H // ratio, W // ratio))
        x = x.flatten(0, 1).contiguous()
        # 阶段 2：conv -> LIF -> pool（pooling_stat[2]=='1'，LIF 在池化之前）
        x = self.proj_conv2(x)
        x = self.lif2(x.reshape(T, B, -1, H // ratio, W // ratio))
        x = x.flatten(0, 1).contiguous()
        x = self.maxpool(x)
        ratio *= 2
        # 阶段 3：conv -> pool -> x_feat -> LIF（pooling_stat[3]=='1'，池化在 LIF 之前）
        x = self.proj_conv3(x)
        x = self.maxpool(x)
        ratio *= 2
        x_feat = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.lif3(x_feat)
        x = x.flatten(0, 1).contiguous()
        # rpe 分支：conv 后加回 LIF 之前的 x_feat（无 LIF，与原模型 forward 一致）
        x = (self.rpe_conv(x) + x_feat.flatten(0, 1))
        return x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()


class FusedSSA(nn.Module):
    """与 module/ms_conv.py MS_SSA_Conv 逐算子同构的融合版 SSA（direct_xor 注意力）。

    差异：q/k/v/proj 的 Conv+BN 融合为单个 Conv(bias=True)；talking_heads Conv1d
    在原模型 forward 中未被调用（只有 talking_heads_lif），故此处不建该层。
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.k_conv = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.v_conv = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_conv = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        # 注意力/头混合 LIF：v_threshold=0.5（与原模型一致）
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True,
                                         backend="torch")
        self.talking_heads_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5,
                                                  detach_reset=True, backend="torch")
        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

    def forward(self, x):
        """输入/输出均为 [T,B,C,H,W]；返回 (输出, v) 与原 SSA forward 对齐。"""
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        # shortcut LIF -> q/k/v 三分支（conv -> LIF）
        x = self.shortcut_lif(x)
        x_for_qkv = x.flatten(0, 1)
        q_lif = self.q_lif(self.q_conv(x_for_qkv).reshape(T, B, C, H, W).contiguous())
        k_lif = self.k_lif(self.k_conv(x_for_qkv).reshape(T, B, C, H, W).contiguous())
        v_lif = self.v_lif(self.v_conv(x_for_qkv).reshape(T, B, C, H, W).contiguous())
        # 头维重排：[T,B,C,H,W] -> [T,B,heads,N,head_dim]
        q = (q_lif.flatten(3).transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())
        k = (k_lif.flatten(3).transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())
        v = (v_lif.flatten(3).transpose(-1, -2)
             .reshape(T, B, N, self.num_heads, C // self.num_heads)
             .permute(0, 1, 3, 2, 4).contiguous())
        # kv = k ⊙ v 按 token 维求和 -> talking_heads_lif（v_threshold=0.5）
        kv = k.mul(v).sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        # q ⊙ kv（广播） -> 还原 [T,B,C,H,W]
        x = q.mul(kv).transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        # proj conv（融合版无 BN）-> 残差
        x = (self.proj_conv(x.flatten(0, 1))
             .reshape(T, B, C, H, W).contiguous())
        return x + identity, v


class FusedMLP(nn.Module):
    """与 module/ms_conv.py MS_MLP_Conv 逐算子同构的融合版 MLP。

    差异：fc1/fc2 的 Conv+BN 融合为单个 Conv(bias=True)。mlp_ratio=4 时
    in_features != hidden_features，原模型的 fc1 残差不触发，故省略。
    """

    def __init__(self, dim: int, mlp_ratio: int):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1_conv = nn.Conv2d(dim, hidden, 1, 1, bias=True)
        self.fc2_conv = nn.Conv2d(hidden, dim, 1, 1, bias=True)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

    def forward(self, x):
        """输入/输出均为 [T,B,C,H,W]；identity 取入口值（与原模型 res=False 路径一致）。"""
        T, B, _, H, W = x.shape
        identity = x
        # fc1：LIF -> conv（融合版无 BN）-> reshape 回 [T,B,hidden,H,W]
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        # fc2：LIF -> conv -> reshape 回 [T,B,C,H,W] -> 残差
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        return x + identity


class FusedBlock(nn.Module):
    """融合版 block：SSA + MLP（与 MS_Block_Conv 顺序一致）。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.attn = FusedSSA(dim, num_heads)
        self.mlp = FusedMLP(dim, mlp_ratio)

    def forward(self, x):
        """输入/输出均为 [T,B,C,H,W]。"""
        x, _ = self.attn(x)
        return self.mlp(x)


class FusedSDT(nn.Module):
    """与 Burn 端（BN 已融合）严格同构的 SDT 推理形式，可直接训练。

    结构：FusedSPS -> depths x FusedBlock -> flatten(3).mean(3) -> head_lif
          -> nn.Linear -> mean(0)。
    LIF 均为 spikingjelly torch 后端（每次 forward 前膜电位清零，eval/train 语义一致）。
    """

    def __init__(self, img_h=32, img_w=32, in_ch=3, num_classes=10, embed_dims=256,
                 num_heads=8, mlp_ratio=4, depths=2, T=4):
        super().__init__()
        self.T = T
        self.embed_dims = embed_dims
        self.patch_embed = FusedSPS(img_h, img_w, in_ch, embed_dims)
        self.blocks = nn.ModuleList(
            [FusedBlock(embed_dims, num_heads, mlp_ratio) for _ in range(depths)]
        )
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.head = nn.Linear(embed_dims, num_classes)

    def forward(self, x):
        """支持 [B,3,32,32]（自动 repeat 成 T 步）或 [T,B,3,32,32] 输入。

        返回 (logits_mean [B,num_classes], logits_all [T,B,num_classes])，
        后者仅供 --validate-weights 逐算子对照使用。
        """
        if len(x.shape) < 5:
            # 单帧输入 -> 沿时间维复制 T 份（与 SpikeDrivenTransformer.forward 一致）
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # patch embed
        x = self.patch_embed(x)
        # transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # token 求均值 -> head_lif -> 线性头 -> 时间步平均
        x = x.flatten(3).mean(3)
        x = self.head_lif(x)
        logits = self.head(x)
        return logits.mean(0), logits


# ---------------------------------------------------------------------------
# 权重加载 / 导出
# ---------------------------------------------------------------------------

def load_fused_weights(model: FusedSDT, npz_path: str) -> None:
    """从 export_reference.py 生成的 NPZ 加载「BN 融合形式」权重到 FusedSDT。

    NPZ 中的 W'/B' 是 Conv+BN 融合结果（W' = w*g/sqrt(var+eps)），
    融合形式下 Conv(w', b') 与 Conv+BN 推理严格等价，故直接赋给 Conv 的
    weight/bias，无需拆回 BN（mu=0/var=1 拆法会破坏权重分布，且原 q/k/v
    conv 无 bias 无法承载 B'，故采用整体前向等价方案）。
    """
    with np.load(npz_path) as data:
        pe = model.patch_embed
        # patch embed 4 级 conv + rpe conv（bias=True，B' 即融合偏置）
        pe_w = ["pe_proj0_w", "pe_proj1_w", "pe_proj2_w", "pe_proj3_w", "pe_rpe_w"]
        pe_b = ["pe_proj0_b", "pe_proj1_b", "pe_proj2_b", "pe_proj3_b", "pe_rpe_b"]
        for name, w_key, b_key in zip(
            ["proj_conv0", "proj_conv1", "proj_conv2", "proj_conv3", "rpe_conv"],
            pe_w, pe_b,
        ):
            conv = getattr(pe, name)
            conv.weight.data.copy_(torch.from_numpy(data[w_key]).float())
            conv.bias.data.copy_(torch.from_numpy(data[b_key]).float())
        # 每个 block：q/k/v/proj/fc1/fc2 conv 融合权重 + 头
        for j, blk in enumerate(model.blocks):
            convs = {
                "q_conv": (f"blk{j}_q_w", f"blk{j}_q_b"),
                "k_conv": (f"blk{j}_k_w", f"blk{j}_k_b"),
                "v_conv": (f"blk{j}_v_w", f"blk{j}_v_b"),
                "proj_conv": (f"blk{j}_proj_w", f"blk{j}_proj_b"),
            }
            for name, (w_key, b_key) in convs.items():
                conv = getattr(blk.attn, name)
                conv.weight.data.copy_(torch.from_numpy(data[w_key]).float())
                conv.bias.data.copy_(torch.from_numpy(data[b_key]).float())
            blk.mlp.fc1_conv.weight.data.copy_(
                torch.from_numpy(data[f"blk{j}_fc1_w"]).float())
            blk.mlp.fc1_conv.bias.data.copy_(
                torch.from_numpy(data[f"blk{j}_fc1_b"]).float())
            blk.mlp.fc2_conv.weight.data.copy_(
                torch.from_numpy(data[f"blk{j}_fc2_w"]).float())
            blk.mlp.fc2_conv.bias.data.copy_(
                torch.from_numpy(data[f"blk{j}_fc2_b"]).float())
        # 分类头（Linear）
        model.head.weight.data.copy_(torch.from_numpy(data["head_w"]).float())
        model.head.bias.data.copy_(torch.from_numpy(data["head_b"]).float())


def calibrate_fused_weights(model: FusedSDT, calib_images: np.ndarray,
                            device: torch.device, num_batches: int = 16,
                            max_rounds: int = 10, tol: float = 1e-3) -> None:
    """逐层静态校准融合权重：等价于「用真实数据统计量重新融合 BN」。

    背景：NPZ 融合权重基于初始 BN running stats（mu=0, var=1），而真实激活方差
    极小（首层 conv 输出幅值 ~0.03，LIF 阈值 1.0），直接训练会全网零脉冲死锁。
    校准对每个「输出直接喂给 LIF 的层」测量其输出每通道 (mu, sigma)，再做
        W' <- W'/sigma,  B' <- (B' - mu)/sigma
    的等效替换（该层输出被标准化为均值 0、方差 1，与原模型 train 模式下 BN 的
    作用一致）。注意校准只改数值、不改网络形式，Burn 端同构性不受影响。

    由于层间级联依赖（上游 LIF 激活后下游 conv 输出统计才会变化），采用迭代
    校准直到各层 sigma 变化收敛（通常 2~4 轮）。
    """
    # 前向路径上的 (模块引用, 键名) 列表（顺序与 forward 一致，便于日志展示）
    pe = model.patch_embed
    layer_list = [
        (pe.proj_conv0, "pe_proj0"), (pe.proj_conv1, "pe_proj1"),
        (pe.proj_conv2, "pe_proj2"), (pe.proj_conv3, "pe_proj3"),
        (pe.rpe_conv, "pe_rpe"),
    ]
    for j, blk in enumerate(model.blocks):
        a, m = blk.attn, blk.mlp
        layer_list += [
            (a.q_conv, f"blk{j}_q"), (a.k_conv, f"blk{j}_k"),
            (a.v_conv, f"blk{j}_v"), (a.proj_conv, f"blk{j}_proj"),
            (m.fc1_conv, f"blk{j}_fc1"), (m.fc2_conv, f"blk{j}_fc2"),
        ]
    layer_list.append((model.head, "head"))

    # 校准子集（取训练子集前若干 batch，固定顺序无随机性）
    xs = torch.from_numpy(calib_images[:num_batches * 32]).float().to(device)

    for round_idx in range(1, max_rounds + 1):
        report = []
        for mod, name in layer_list:
            # 钩子收集该层输出，统一转成 [N, C, ...] 布局（conv 输出已是 [TB,C,H,W]；
            # head 的 Linear 输出是 [T,B,C]，需转置时间维后合并）
            collected = []

            def tap(m, inp, out, _c=collected):
                o = out.detach()
                if o.dim() == 3:  # [T,B,C] -> [B*T, C]
                    o = o.transpose(0, 1).reshape(-1, o.shape[2])
                _c.append(o)

            handle = mod.register_forward_hook(tap)
            with torch.no_grad():
                for i in range(num_batches):
                    functional.reset_net(model)
                    model(xs[i * 32:(i + 1) * 32])
            handle.remove()
            out = torch.cat(collected, dim=0)  # [N*T*B, C, ...]
            # 每通道统计（对通道维之外的其余维求统计量）
            dims = [0] + list(range(2, out.dim()))
            mu = out.mean(dim=dims)
            var = out.var(dim=dims, unbiased=False)
            sigma = torch.sqrt(var + 1e-6)
            # 等效替换：W' / sigma, (B' - mu) / sigma
            # conv 权重 [out,in,kH,kW] 按通道维缩放；Linear 权重 [out,in] 按输出维缩放
            if mod.weight.dim() == 4:
                mod.weight.data.div_(sigma.reshape(-1, 1, 1, 1))
            else:
                mod.weight.data.div_(sigma.reshape(-1, 1))
            mod.bias.data.sub_(mu).div_(sigma)
            report.append((name, float(sigma.mean())))
        # 收敛判断：各层平均 sigma 均接近 1（说明统计量已稳定）
        err = max(abs(s - 1.0) for _, s in report)
        print(f"[calibrate] 第 {round_idx} 轮: max|sigma-1|={err:.4f} "
              f"（层平均 sigma 范围 "
              f"{min(s for _, s in report):.3f}~{max(s for _, s in report):.3f}）")
        if err < tol:
            print(f"[calibrate] 已收敛（{round_idx} 轮）")
            break


def save_fused_weights(model: FusedSDT, out_path: str) -> None:
    """把 FusedSDT 初始权重按 NPZ 键名约定导出（供后续复现/对照）。"""
    out = {}
    pe = model.patch_embed
    # patch embed 4 级 conv + rpe conv
    for name, w_key, b_key in zip(
        ["proj_conv0", "proj_conv1", "proj_conv2", "proj_conv3", "rpe_conv"],
        ["pe_proj0_w", "pe_proj1_w", "pe_proj2_w", "pe_proj3_w", "pe_rpe_w"],
        ["pe_proj0_b", "pe_proj1_b", "pe_proj2_b", "pe_proj3_b", "pe_rpe_b"],
    ):
        conv = getattr(pe, name)
        out[w_key] = conv.weight.detach().cpu().numpy().astype(np.float32)
        out[b_key] = conv.bias.detach().cpu().numpy().astype(np.float32)
    # 每个 block
    for j, blk in enumerate(model.blocks):
        convs = {
            "q_conv": (f"blk{j}_q_w", f"blk{j}_q_b"),
            "k_conv": (f"blk{j}_k_w", f"blk{j}_k_b"),
            "v_conv": (f"blk{j}_v_w", f"blk{j}_v_b"),
            "proj_conv": (f"blk{j}_proj_w", f"blk{j}_proj_b"),
        }
        for name, (w_key, b_key) in convs.items():
            conv = getattr(blk.attn, name)
            out[w_key] = conv.weight.detach().cpu().numpy().astype(np.float32)
            out[b_key] = conv.bias.detach().cpu().numpy().astype(np.float32)
        out[f"blk{j}_fc1_w"] = blk.mlp.fc1_conv.weight.detach().cpu().numpy().astype(
            np.float32)
        out[f"blk{j}_fc1_b"] = blk.mlp.fc1_conv.bias.detach().cpu().numpy().astype(
            np.float32)
        out[f"blk{j}_fc2_w"] = blk.mlp.fc2_conv.weight.detach().cpu().numpy().astype(
            np.float32)
        out[f"blk{j}_fc2_b"] = blk.mlp.fc2_conv.bias.detach().cpu().numpy().astype(
            np.float32)
    # 分类头
    out["head_w"] = model.head.weight.detach().cpu().numpy().astype(np.float32)
    out["head_b"] = model.head.bias.detach().cpu().numpy().astype(np.float32)
    # 元信息（与 export_reference.py 对齐）
    out["meta"] = np.array([32, 16, 3, 10, 256, 8, 4, 2, 4], dtype=np.int64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **out)
    print(f"[save-init] 已导出初始权重: {out_path}")


def validate_weights(model: FusedSDT, npz_path: str, device: torch.device) -> None:
    """用 NPZ 中记录的固定输入/中间量逐算子对照 FusedSDT 前向，验证同构性。"""
    with np.load(npz_path) as data:
        # 1) 固定输入（[T,B,3,32,32]）前向，对照最终 logits
        x = torch.from_numpy(data["input"]).float().to(device)
        logits_ref = data["logits"]
        # LIF 推理为无状态（forward 前膜电位清零），no_grad+eval 下直接调用即可
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
        diff = float(np.abs(logits.cpu().numpy() - logits_ref).max())
        print(f"[validate-weights] 最大绝对误差(logits): {diff:.3e}")
        assert diff < 1e-3, "FusedSDT 与 export_reference 的 logits 偏差过大"

        # 2) 逐算子中间量对照（SPS/block/head 各关键步）
        model_patch = model.patch_embed
        xs = x
        T, B, _, H, W = xs.shape
        ratio = 1
        # SPS 阶段 0/1/2（与 NPZ 记录的 LIF 输出对照）
        x = model_patch.lif0(model_patch.proj_conv0(xs.flatten(0, 1))
                             .reshape(T, B, -1, H // ratio, W // ratio))
        diff = float(np.abs(x.detach().cpu().numpy() - data["sps_proj_lif"]).max())
        print(f"[validate-weights] sps_proj_lif 最大绝对误差: {diff:.3e}")
        x = x.flatten(0, 1)
        x = model_patch.lif1(model_patch.proj_conv1(x)
                             .reshape(T, B, -1, H // ratio, W // ratio))
        diff = float(np.abs(x.detach().cpu().numpy() - data["sps_proj1_lif"]).max())
        print(f"[validate-weights] sps_proj1_lif 最大绝对误差: {diff:.3e}")
        x = x.flatten(0, 1)
        x = model_patch.lif2(model_patch.proj_conv2(x)
                             .reshape(T, B, -1, H // ratio, W // ratio))
        diff = float(np.abs(x.detach().cpu().numpy() - data["sps_proj2_lif"]).max())
        print(f"[validate-weights] sps_proj2_lif 最大绝对误差: {diff:.3e}")
        x = model_patch.maxpool(x.flatten(0, 1))
        x = model_patch.proj_conv3(x)
        x = model_patch.maxpool(x)
        ratio = 4
        x_feat = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        diff = float(np.abs(x_feat.detach().cpu().numpy() - data["sps_x_feat"]).max())
        print(f"[validate-weights] sps_x_feat 最大绝对误差: {diff:.3e}")
        x = model_patch.lif3(x_feat).flatten(0, 1)
        x = (model_patch.rpe_conv(x) + x_feat.flatten(0, 1))
        x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        diff = float(np.abs(x.detach().cpu().numpy() - data["sps_rpe_out"]).max())
        print(f"[validate-weights] sps_rpe_out 最大绝对误差: {diff:.3e}")

        # 3) 每个 block 的 SSA/MLP 中间量对照
        for j, blk in enumerate(model.blocks):
            a = blk.attn
            # 块级特征图尺寸（与输入图像无关）：[T,B,C,Hb,Wb]，此处 C=embed_dims
            _, _, Cb, Hb, Wb = x.shape
            identity = x
            x_short = a.shortcut_lif(x)
            diff = float(np.abs(x_short.detach().cpu().numpy()
                                - data[f"blk{j}_shortcut_lif"]).max())
            print(f"[validate-weights] blk{j}_shortcut_lif 最大绝对误差: {diff:.3e}")
            # q/k/v 分支：conv（4 维输入）输出 reshape 回 [T,B,C,Hb,Wb] 后过 LIF
            q_lif = a.q_lif(a.q_conv(x_short.flatten(0, 1))
                            .reshape(T, B, Cb, Hb, Wb))
            diff = float(np.abs(q_lif.detach().cpu().numpy()
                                - data[f"blk{j}_q_lif"]).max())
            print(f"[validate-weights] blk{j}_q_lif 最大绝对误差: {diff:.3e}")
            k_lif = a.k_lif(a.k_conv(x_short.flatten(0, 1))
                            .reshape(T, B, Cb, Hb, Wb))
            diff = float(np.abs(k_lif.detach().cpu().numpy()
                                - data[f"blk{j}_k_lif"]).max())
            print(f"[validate-weights] blk{j}_k_lif 最大绝对误差: {diff:.3e}")
            v_lif = a.v_lif(a.v_conv(x_short.flatten(0, 1))
                            .reshape(T, B, Cb, Hb, Wb))
            diff = float(np.abs(v_lif.detach().cpu().numpy()
                                - data[f"blk{j}_v_lif"]).max())
            print(f"[validate-weights] blk{j}_v_lif 最大绝对误差: {diff:.3e}")
            # q/k/v 头维重排 -> kv 求和 -> talking_heads_lif -> q⊙kv
            N = Hb * Wb
            q = (q_lif.flatten(3).transpose(-1, -2)
                 .reshape(T, B, N, a.num_heads, Cb // a.num_heads)
                 .permute(0, 1, 3, 2, 4).contiguous())
            k = (k_lif.flatten(3).transpose(-1, -2)
                 .reshape(T, B, N, a.num_heads, Cb // a.num_heads)
                 .permute(0, 1, 3, 2, 4).contiguous())
            v = (v_lif.flatten(3).transpose(-1, -2)
                 .reshape(T, B, N, a.num_heads, Cb // a.num_heads)
                 .permute(0, 1, 3, 2, 4).contiguous())
            kv_sum = k.mul(v).sum(dim=-2, keepdim=True)
            diff = float(np.abs(kv_sum.detach().cpu().numpy()
                                - data[f"blk{j}_kv_sum"]).max())
            print(f"[validate-weights] blk{j}_kv_sum 最大绝对误差: {diff:.3e}")
            kv_lif = a.talking_heads_lif(kv_sum)
            diff = float(np.abs(kv_lif.detach().cpu().numpy()
                                - data[f"blk{j}_kv_lif"]).max())
            print(f"[validate-weights] blk{j}_kv_lif 最大绝对误差: {diff:.3e}")
            x_attn = (q.mul(kv_lif).transpose(3, 4)
                      .reshape(T, B, Cb, Hb, Wb).contiguous())
            diff = float(np.abs(x_attn.detach().cpu().numpy()
                                - data[f"blk{j}_x_attn"]).max())
            print(f"[validate-weights] blk{j}_x_attn 最大绝对误差: {diff:.3e}")
            proj_out = (a.proj_conv(x_attn.flatten(0, 1))
                        .reshape(T, B, Cb, Hb, Wb).contiguous())
            diff = float(np.abs(proj_out.detach().cpu().numpy()
                                - data[f"blk{j}_proj_out"]).max())
            print(f"[validate-weights] blk{j}_proj_out 最大绝对误差: {diff:.3e}")
            ssa_out = proj_out + identity
            diff = float(np.abs(ssa_out.detach().cpu().numpy()
                                - data[f"blk{j}_ssa_out"]).max())
            print(f"[validate-weights] blk{j}_ssa_out 最大绝对误差: {diff:.3e}")
            # MLP：fc1_lif -> conv -> fc2_lif -> conv -> 残差
            m = blk.mlp
            fc1_lif = m.fc1_lif(ssa_out)
            diff = float(np.abs(fc1_lif.detach().cpu().numpy()
                                - data[f"blk{j}_mlp_fc1_lif"]).max())
            print(f"[validate-weights] blk{j}_mlp_fc1_lif 最大绝对误差: {diff:.3e}")
            fc1_out = m.fc1_conv(fc1_lif.flatten(0, 1)).reshape(T, B, -1, Hb, Wb)
            diff = float(np.abs(fc1_out.detach().cpu().numpy()
                                - data[f"blk{j}_mlp_fc1_out"]).max())
            print(f"[validate-weights] blk{j}_mlp_fc1_out 最大绝对误差: {diff:.3e}")
            fc2_lif = m.fc2_lif(fc1_out)
            diff = float(np.abs(fc2_lif.detach().cpu().numpy()
                                - data[f"blk{j}_mlp_fc2_lif"]).max())
            print(f"[validate-weights] blk{j}_mlp_fc2_lif 最大绝对误差: {diff:.3e}")
            mlp_out = m.fc2_conv(fc2_lif.flatten(0, 1)).reshape(T, B, Cb, Hb, Wb)
            diff = float(np.abs(mlp_out.detach().cpu().numpy()
                                - data[f"blk{j}_mlp_out"]).max())
            print(f"[validate-weights] blk{j}_mlp_out 最大绝对误差: {diff:.3e}")
            mlp_res = mlp_out + identity
            diff = float(np.abs(mlp_res.detach().cpu().numpy()
                                - data[f"blk{j}_mlp_res"]).max())
            print(f"[validate-weights] blk{j}_mlp_res 最大绝对误差: {diff:.3e}")
            x = mlp_res

        # 4) head：flatten(3).mean(3) -> head_lif -> Linear -> mean(0)
        feat_mean = x.flatten(3).mean(3)
        diff = float(np.abs(feat_mean.detach().cpu().numpy()
                            - data["feat_mean"]).max())
        print(f"[validate-weights] feat_mean 最大绝对误差: {diff:.3e}")
        head_lif_out = model.head_lif(feat_mean)
        diff = float(np.abs(head_lif_out.detach().cpu().numpy()
                            - data["head_lif_out"]).max())
        print(f"[validate-weights] head_lif_out 最大绝对误差: {diff:.3e}")
        logits_all = model.head(head_lif_out)
        logits_mean = logits_all.mean(0)
        diff = float(np.abs(logits_mean.detach().cpu().numpy() - logits_ref).max())
        print(f"[validate-weights] 最终 logits 最大绝对误差: {diff:.3e}")
    print("[validate-weights] 对照完成")


# ---------------------------------------------------------------------------
# 数据：直接使用与 Burn 端完全相同的 NPZ 数据
# ---------------------------------------------------------------------------

class NpzCifar10(Dataset):
    """读取 rust-sdt/artifacts/cifar10_data.npz 的 CIFAR-10 子集数据集。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(np.ascontiguousarray(images)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------

def evaluate(model: FusedSDT, loader: DataLoader, device: torch.device) -> float:
    """在给定数据集上评估 top-1 准确率（%）。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # LIF 无状态语义：每次前向前清零膜电位（与 Burn 端 / export_reference 一致）
            functional.reset_net(model)
            logits, _ = model(imgs)  # 内部自动 repeat 为 T 步
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.shape[0]
    return 100.0 * correct / total


def main() -> None:
    """主入口：解析参数 -> 建模型 -> 加载权重 -> 训练 -> 写 CSV。"""
    parser = argparse.ArgumentParser(description="PyTorch 侧小规模训练对照脚本")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数（默认 2）")
    parser.add_argument("--batch-size", type=int, default=32, help="batch 大小（默认 32）")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率（默认 0.01）")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子（默认 42）")
    parser.add_argument("--init-npz", type=str, default=None,
                        help="可选：从 NPZ 加载融合权重（export_reference.py 生成）")
    parser.add_argument("--out", type=str,
                        default=os.path.join("rust-sdt", "artifacts",
                                             "train_pytorch.csv"),
                        help="训练指标 CSV 输出路径")
    parser.add_argument("--save-init", type=str, default=None,
                        help="可选：导出初始化权重 NPZ 路径")
    parser.add_argument("--validate-weights", action="store_true",
                        help="加载 init-npz 后用固定输入逐算子对照前向")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="跳过静态校准（默认校准：修正初始 BN 统计量与真实激活的失配）")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader 工作进程数（默认 0）")
    args = parser.parse_args()

    # 固定种子（numpy / torch；DataLoader shuffle 用独立 Generator）
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # 设备自动选择：优先 CUDA，无 GPU 时回退 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] 设备: {device}（torch {torch.__version__}）")

    # 读取与 Burn 端完全同一份数据（保证子集一致）
    data_path = os.path.join(ROOT, DEFAULT_DATA_NPZ)
    with np.load(data_path) as data:
        train_ds = NpzCifar10(data["train_x"], data["train_y"])
        test_ds = NpzCifar10(data["test_x"], data["test_y"])
    print(f"[info] 训练集 {len(train_ds)} 样本 / 测试集 {len(test_ds)} 样本 "
          f"（来自 {data_path}）")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=g,
                              drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # 构建融合版模型（与 Burn 端同构）
    model = FusedSDT(img_h=32, img_w=32, in_ch=3, num_classes=NUM_CLASSES,
                     embed_dims=256, num_heads=8, mlp_ratio=4, depths=2,
                     T=T_STEPS).to(device)
    if args.init_npz:
        # init-npz 模式：不执行 trunc_normal 初始化，直接加载融合权重
        init_npz = args.init_npz if os.path.isabs(args.init_npz) \
            else os.path.join(ROOT, args.init_npz)
        load_fused_weights(model, init_npz)
        print(f"[info] 已从 {init_npz} 加载融合权重（无 trunc_normal 初始化）")
        if args.validate_weights:
            validate_weights(model, init_npz, device)
        # 静态校准：修正初始 BN running stats 与真实激活分布的失配（否则零脉冲死锁）
        if not args.no_calibrate:
            calib_images = train_ds.images.numpy()
            calibrate_fused_weights(model, calib_images, device)
    else:
        # 自身初始化：conv 权重 trunc_normal(std=0.02)、bias 清零、head 默认初始化
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        print("[info] 使用 trunc_normal(std=0.02) 随机初始化")
    if args.save_init:
        save_path = args.save_init if os.path.isabs(args.save_init) \
            else os.path.join(ROOT, args.save_init)
        save_fused_weights(model, save_path)

    # 训练组件：SGD(momentum=0.9) + 交叉熵
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 可训练参数量统计
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] 可训练参数量: {n_params}")

    # 初始评估（便于与 Burn 端对照起点）
    t_train_start = time.time()
    acc0 = evaluate(model, test_loader, device)
    print(f"[epoch 0] 初始 val_top1: {acc0:.2f}%")

    # 训练循环：逐 epoch 训练 + 测试集评估 top-1，并写入 CSV
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_sum = 0.0
        n_samples = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # LIF 无状态语义：每次前向前清零膜电位（与 Burn 端 / SDT 官方训练循环一致），
            # 否则跨 batch 的残留膜电位会污染下一前向且导致反向图复用错误
            functional.reset_net(model)
            # 前向（内部自动 repeat 为 T=4 步），取时间平均 logits
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            bs = labels.shape[0]
            loss_sum += loss.item() * bs
            n_samples += bs
        train_loss = loss_sum / n_samples
        val_top1 = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        rows.append((epoch, train_loss, val_top1))
        print(f"[epoch {epoch}/{args.epochs}] train_loss: {train_loss:.6f} | "
              f"val_top1: {val_top1:.2f}% | 本轮耗时 {epoch_time:.1f}s")

    total_time = time.time() - t_train_start
    print(f"[info] 总训练耗时（含初始评估）: {total_time:.1f}s")

    # 写 CSV：表头与格式与 Burn 端完全一致（epoch 从 1，loss 6 位小数，top1 百分数）
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("epoch,train_loss,val_top1\n")
        for ep, tl, ta in rows:
            f.write(f"{ep},{tl:.6f},{ta:.2f}\n")
    print(f"[info] 指标已写入: {out_path}")


if __name__ == "__main__":
    main()
