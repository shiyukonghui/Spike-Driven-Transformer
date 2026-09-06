# -*- coding: utf-8 -*-
"""
PyTorch 参考侧导出脚本：将 SpikeDrivenTransformer 的权重与对照张量导出为 NPZ。

内容：
1. 固定随机种子构建 SDT 模型（dim=256, layer=2, heads=8, T=4, 32x32 输入）；
2. 用固定种子的输入张量执行前向，导出：
   - 输入 x [T,B,C,H,W]
   - 最终 logits [num_classes]（T 平均后）
   - head_lif 前的均值特征 [T,B,embed_dims]
3. 权重以「BN 推理融合」形式导出：Conv(w) + BN(g,b,mu,var) => W' = w*g/sqrt(var+eps),
   B' = b - mu*g/sqrt(var+eps)（conv 无 bias 时 B' 即 bias）。
   导出键名与 Burn 端 loader 一一对应。
4. 逐模块中间张量（手动展开 forward、no_grad + eval 下截获）：SPS 各级 LIF 输出、
   每个 block 的 SSA/MLP 中间量、head_lif 输出；键名与 rust-sdt 端逐层对照约定一致。

运行：PYTHONPATH=.pylibs python scripts/export_reference.py
输出：rust-sdt/artifacts/sdt_reference.npz
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# 保证项目根目录可导入（model/, module/）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from timm.models.layers import trunc_normal_  # noqa: E402
from spikingjelly.clock_driven.neuron import MultiStepLIFNode  # noqa: E402
from spikingjelly.clock_driven import functional  # noqa: E402

# 项目内模型（torch 侧）
from model.spikeformer import SpikeDrivenTransformer  # noqa: E402

# cupy 内核不可用时回退到 torch 后端（数值语义完全一致，仅实现不同）
import spikingjelly.clock_driven.neuron as _sj_neuron  # noqa: E402

if _sj_neuron.cupy is None:
    def _patched_ms_lif(tau, **kw):
        kw.pop("backend", None)
        return _sj_neuron.MultiStepLIFNode(tau=tau, **kw)

    def _patched_ms_plif(init_tau, **kw):
        kw.pop("backend", None)
        return _sj_neuron.MultiStepParametricLIFNode(init_tau=init_tau, **kw)

    _sj_neuron.MultiStepLIFNode.__init__.__defaults__ = (
        2.0, True, 1.0, 0.0, None, False, "torch", 64,
    )
    # 直接替换项目模块引用的构造调用：monkeypatch check_backend + backend 默认值
    _orig_init = _sj_neuron.MultiStepLIFNode.__init__

    def _lif_init(self, tau=2.0, decay_input=True, v_threshold=1.0, v_reset=0.0,
                  surrogate_function=None, detach_reset=False, backend="torch", lava_s_cale=64):
        if surrogate_function is None:
            surrogate_function = _sj_neuron.surrogate.Sigmoid()
        _orig_init(self, tau, decay_input, v_threshold, v_reset,
                   surrogate_function, detach_reset, "torch", lava_s_cale)

    _sj_neuron.MultiStepLIFNode.__init__ = _lif_init


def set_seed(seed: int):
    """固定全局随机种子，保证权重初始化与输入完全可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bn_fused(conv_w, conv_b, bn):
    """将 Conv2d 权重与 BatchNorm2d 融合为单个 (W', B')。

    BN 推理：y = (x - mu) / sqrt(var + eps) * g + b
    融合：  W' = W * g / sqrt(var + eps)
            B' = (conv_b - mu) * g / sqrt(var + eps) + b
    """
    g = bn.weight.detach()
    b = bn.bias.detach()
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    scale = g / torch.sqrt(var + eps)
    w = conv_w.detach() * scale.reshape(-1, 1, 1, 1)
    if conv_b is None:
        bias = b - mu * scale
    else:
        bias = (conv_b.detach() - mu) * scale + b
    return w, bias


def main():
    seed = 42
    set_seed(seed)

    # 与 conf/cifar10/2_256_300E_t4.yml 对齐的模型配置
    model = SpikeDrivenTransformer(
        img_size_h=32,
        img_size_w=32,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dims=256,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        depths=2,
        sr_ratios=[1],
        T=4,
        pooling_stat="0011",
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs_mode=False,
        TET=False,
    )
    model.eval()  # eval 模式：BN 用 running stats；DropPath/Dropout 关闭

    # 固定种子的输入：时间维 T=4，batch B=2，3x32x32
    g = torch.Generator().manual_seed(20240601)
    x = torch.rand(4, 2, 3, 32, 32, generator=g)
    # 每个时间步给一点差异，避免全同输入掩盖时间维 bug
    x = x + 0.1 * torch.randn(4, 2, 3, 32, 32, generator=g)

    with torch.no_grad():
        # 手动展开 forward 完整链路并截获逐模块中间张量。
        # 说明：spikingjelly 的 MultiStepLIFNode 推理时无状态（每次 forward 前 v=0），
        # 因此直接依次调用各 LIF 即可；eval 模式下 BN 使用 running stats；
        # forward 中 transpose(0,1) 对 5 维输入为恒等操作，x 已是 [T,B,3,32,32]。
        xs = x
        functional.reset_net(model)  # 确保所有 LIF 状态清零（v=0）

        # ---------- SPS（module/sps.py MS_SPS.forward 手动展开，pooling_stat="0011"） ----------
        pe = model.patch_embed
        T, B = xs.shape[0], xs.shape[1]
        H_img, W_img = xs.shape[3], xs.shape[4]
        ratio = 1
        # 阶段 0：proj_conv + proj_bn + proj_lif（"0"：该级不池化）
        tt = pe.proj_conv(xs.flatten(0, 1))
        tt = pe.proj_bn(tt).reshape(T, B, -1, H_img // ratio, W_img // ratio).contiguous()
        sps_proj_lif = pe.proj_lif(tt)
        tt = sps_proj_lif.flatten(0, 1).contiguous()
        # 阶段 1：proj_conv1 + proj_bn1 + proj_lif1（"0"：该级不池化）
        tt = pe.proj_conv1(tt)
        tt = pe.proj_bn1(tt).reshape(T, B, -1, H_img // ratio, W_img // ratio).contiguous()
        sps_proj1_lif = pe.proj_lif1(tt)
        tt = sps_proj1_lif.flatten(0, 1).contiguous()
        # 阶段 2：proj_conv2 + proj_bn2 + proj_lif2 + maxpool2（"1"：LIF 在池化之前）
        tt = pe.proj_conv2(tt)
        tt = pe.proj_bn2(tt).reshape(T, B, -1, H_img // ratio, W_img // ratio).contiguous()
        sps_proj2_lif = pe.proj_lif2(tt)
        tt = sps_proj2_lif.flatten(0, 1).contiguous()
        tt = pe.maxpool2(tt)
        # 捕获 maxpool2 输出（此时空间 32->16，ratio 尚未更新故用 ratio*2）
        sps_proj2_pool = tt.reshape(T, B, -1, H_img // (ratio * 2), W_img // (ratio * 2)).contiguous()
        ratio *= 2
        # 阶段 3：proj_conv3 + proj_bn3 + maxpool3 -> x_feat（proj_lif3 之前的特征）
        tt = pe.proj_conv3(tt)
        tt = pe.proj_bn3(tt)
        tt = pe.maxpool3(tt)
        ratio *= 2
        sps_x_feat = tt.reshape(T, B, -1, H_img // ratio, W_img // ratio).contiguous()
        # proj_lif3 输出
        sps_proj3_lif = pe.proj_lif3(sps_x_feat)
        # rpe 分支：rpe_conv + rpe_bn 后加回 x_feat（注意残差加的是 LIF 前的 x_feat）
        tt = sps_proj3_lif.flatten(0, 1).contiguous()
        tt = pe.rpe_conv(tt)
        tt = pe.rpe_bn(tt)
        sps_rpe_out = (tt + sps_x_feat.flatten(0, 1)).reshape(
            T, B, -1, H_img // ratio, W_img // ratio
        ).contiguous()
        cur = sps_rpe_out

        # ---------- 每个 block：SSA（MS_SSA_Conv）+ MLP（MS_MLP_Conv）手动展开 ----------
        mid_blocks = []  # 收集每个 block 的中间张量，键名去掉 blk{j}_ 前缀
        for j, blk in enumerate(model.block):
            a = blk.attn
            Tt, Bt, C, H, W = cur.shape
            identity = cur
            # shortcut LIF
            blk_shortcut_lif = a.shortcut_lif(cur)
            x_for_qkv = blk_shortcut_lif.flatten(0, 1)
            # q 分支：q_conv + q_bn + q_lif
            tt = a.q_bn(a.q_conv(x_for_qkv)).reshape(Tt, Bt, C, H, W).contiguous()
            blk_q_lif = a.q_lif(tt)
            # k 分支：k_conv + k_bn + k_lif
            tt = a.k_bn(a.k_conv(x_for_qkv)).reshape(Tt, Bt, C, H, W).contiguous()
            blk_k_lif = a.k_lif(tt)
            # v 分支：v_conv + v_bn + v_lif
            tt = a.v_bn(a.v_conv(x_for_qkv)).reshape(Tt, Bt, C, H, W).contiguous()
            blk_v_lif = a.v_lif(tt)
            # 头维重排：[T,B,C,H,W] -> [T,B,heads,N,head_dim]
            N = H * W
            q = (
                blk_q_lif.flatten(3).transpose(-1, -2)
                .reshape(Tt, Bt, N, a.num_heads, C // a.num_heads)
                .permute(0, 1, 3, 2, 4).contiguous()
            )
            k = (
                blk_k_lif.flatten(3).transpose(-1, -2)
                .reshape(Tt, Bt, N, a.num_heads, C // a.num_heads)
                .permute(0, 1, 3, 2, 4).contiguous()
            )
            v = (
                blk_v_lif.flatten(3).transpose(-1, -2)
                .reshape(Tt, Bt, N, a.num_heads, C // a.num_heads)
                .permute(0, 1, 3, 2, 4).contiguous()
            )
            # kv = k ⊙ v 后按 token 维求和（talking_heads_lif 之前）
            # 注意：PyTorch 侧 forward 只经过 talking_heads_lif，未调用 talking_heads Conv1d
            blk_kv_sum = k.mul(v).sum(dim=-2, keepdim=True)
            blk_kv_lif = a.talking_heads_lif(blk_kv_sum)
            # x_attn = q ⊙ kv（广播） -> 还原 [T,B,C,H,W]，即 proj conv 的输入
            blk_x_attn = (
                q.mul(blk_kv_lif).transpose(3, 4).reshape(Tt, Bt, C, H, W).contiguous()
            )
            # proj_conv + proj_bn（残差相加前）
            blk_proj_out = (
                a.proj_bn(a.proj_conv(blk_x_attn.flatten(0, 1)))
                .reshape(Tt, Bt, C, H, W).contiguous()
            )
            # SSA 输出：残差相加（identity 为 shortcut LIF 之前的输入）
            blk_ssa_out = blk_proj_out + identity
            # ---------- MLP 手动展开 ----------
            m = blk.mlp
            m_identity = blk_ssa_out
            blk_mlp_fc1_lif = m.fc1_lif(blk_ssa_out)
            tt = (
                m.fc1_bn(m.fc1_conv(blk_mlp_fc1_lif.flatten(0, 1)))
                .reshape(Tt, Bt, m.c_hidden, H, W).contiguous()
            )
            if m.res:
                # in_features == hidden_features 时存在第一处残差（本配置 mlp_ratio=4 不触发）
                tt = m_identity + tt
                m_identity = tt
            blk_mlp_fc1_out = tt
            blk_mlp_fc2_lif = m.fc2_lif(blk_mlp_fc1_out)
            blk_mlp_out = (
                m.fc2_bn(m.fc2_conv(blk_mlp_fc2_lif.flatten(0, 1)))
                .reshape(Tt, Bt, C, H, W).contiguous()
            )
            # MLP 最终残差相加
            blk_mlp_res = blk_mlp_out + m_identity
            cur = blk_mlp_res
            # 登记 block 级中间张量
            mid_blocks.append({
                "shortcut_lif": blk_shortcut_lif,
                "q_lif": blk_q_lif,
                "k_lif": blk_k_lif,
                "v_lif": blk_v_lif,
                "kv_sum": blk_kv_sum,
                "kv_lif": blk_kv_lif,
                "x_attn": blk_x_attn,
                "proj_out": blk_proj_out,
                "ssa_out": blk_ssa_out,
                "mlp_fc1_lif": blk_mlp_fc1_lif,
                "mlp_fc1_out": blk_mlp_fc1_out,
                "mlp_fc2_lif": blk_mlp_fc2_lif,
                "mlp_out": blk_mlp_out,
                "mlp_res": blk_mlp_res,
            })

        # ---------- head：flatten(3).mean(3) -> head_lif -> Linear ----------
        feat_mean = cur.flatten(3).mean(3)  # [T, B, C]
        head_lif_out = model.head_lif(feat_mean)  # head_lif（无状态、初始 v=0）
        logits = model.head(head_lif_out)  # [T, B, num_classes]
        logits_mean = logits.mean(0)  # [B, num_classes]

    state = model.state_dict()

    out = {}

    # ---------- 输入/输出对照张量 ----------
    out["input"] = x.numpy().astype(np.float32)
    out["feat_mean"] = feat_mean.numpy().astype(np.float32)
    out["logits"] = logits_mean.numpy().astype(np.float32)

    # ---------- 逐模块中间张量（手动展开截获，no_grad + eval） ----------
    # SPS 各级：键名与 Burn 端逐层对照约定一一对应
    out["sps_proj_lif"] = sps_proj_lif.numpy().astype(np.float32)      # [T,B,32,32,32]
    out["sps_proj1_lif"] = sps_proj1_lif.numpy().astype(np.float32)    # [T,B,64,32,32]
    out["sps_proj2_lif"] = sps_proj2_lif.numpy().astype(np.float32)    # [T,B,128,32,32]（LIF 在 maxpool2 之前）
    out["sps_proj2_pool"] = sps_proj2_pool.numpy().astype(np.float32)  # [T,B,128,16,16]（maxpool2 输出，额外对照键）
    out["sps_x_feat"] = sps_x_feat.numpy().astype(np.float32)          # [T,B,256,8,8]（proj_lif3 前）
    out["sps_proj3_lif"] = sps_proj3_lif.numpy().astype(np.float32)    # [T,B,256,8,8]
    out["sps_rpe_out"] = sps_rpe_out.numpy().astype(np.float32)        # [T,B,256,8,8]

    # 每个 block 的 SSA / MLP 中间张量（键名去掉局部变量前缀 blk{j}_，写入时统一拼接）
    for j, mids in enumerate(mid_blocks):
        out[f"blk{j}_shortcut_lif"] = mids["shortcut_lif"].numpy().astype(np.float32)  # [T,B,256,8,8]
        out[f"blk{j}_q_lif"] = mids["q_lif"].numpy().astype(np.float32)                # [T,B,256,8,8]
        out[f"blk{j}_k_lif"] = mids["k_lif"].numpy().astype(np.float32)                # [T,B,256,8,8]
        out[f"blk{j}_v_lif"] = mids["v_lif"].numpy().astype(np.float32)                # [T,B,256,8,8]
        out[f"blk{j}_kv_sum"] = mids["kv_sum"].numpy().astype(np.float32)              # [T,B,8,1,32]
        out[f"blk{j}_kv_lif"] = mids["kv_lif"].numpy().astype(np.float32)              # [T,B,8,1,32]
        out[f"blk{j}_x_attn"] = mids["x_attn"].numpy().astype(np.float32)              # [T,B,256,8,8]
        out[f"blk{j}_proj_out"] = mids["proj_out"].numpy().astype(np.float32)          # [T,B,256,8,8]
        out[f"blk{j}_ssa_out"] = mids["ssa_out"].numpy().astype(np.float32)            # [T,B,256,8,8]
        out[f"blk{j}_mlp_fc1_lif"] = mids["mlp_fc1_lif"].numpy().astype(np.float32)    # [T,B,256,8,8]
        out[f"blk{j}_mlp_fc1_out"] = mids["mlp_fc1_out"].numpy().astype(np.float32)    # [T,B,1024,8,8]
        out[f"blk{j}_mlp_fc2_lif"] = mids["mlp_fc2_lif"].numpy().astype(np.float32)    # [T,B,1024,8,8]
        out[f"blk{j}_mlp_out"] = mids["mlp_out"].numpy().astype(np.float32)            # [T,B,256,8,8]
        out[f"blk{j}_mlp_res"] = mids["mlp_res"].numpy().astype(np.float32)            # [T,B,256,8,8]

    # head_lif 输出（feat_mean 过 head_lif 之后）
    out["head_lif_out"] = head_lif_out.numpy().astype(np.float32)  # [T,B,256]

    # ---------- patch_embed（MS_SPS）权重（BN 融合） ----------
    pe = model.patch_embed

    def dump_stage(prefix, conv, bn):
        w, b = bn_fused(conv.weight, conv.bias, bn)
        out[prefix + "_w"] = w.numpy().astype(np.float32)
        out[prefix + "_b"] = b.numpy().astype(np.float32)

    dump_stage("pe_proj0", pe.proj_conv, pe.proj_bn)
    dump_stage("pe_proj1", pe.proj_conv1, pe.proj_bn1)
    dump_stage("pe_proj2", pe.proj_conv2, pe.proj_bn2)
    dump_stage("pe_proj3", pe.proj_conv3, pe.proj_bn3)
    dump_stage("pe_rpe", pe.rpe_conv, pe.rpe_bn)

    # ---------- 每个 block 的 SSA / MLP 权重（BN 融合） ----------
    for j, blk in enumerate(model.block):
        a = blk.attn
        w, b = bn_fused(a.q_conv.weight, None, a.q_bn)
        out[f"blk{j}_q_w"] = w.numpy().astype(np.float32)
        out[f"blk{j}_q_b"] = b.numpy().astype(np.float32)

        w, b = bn_fused(a.k_conv.weight, None, a.k_bn)
        out[f"blk{j}_k_w"] = w.numpy().astype(np.float32)
        out[f"blk{j}_k_b"] = b.numpy().astype(np.float32)

        w, b = bn_fused(a.v_conv.weight, None, a.v_bn)
        out[f"blk{j}_v_w"] = w.numpy().astype(np.float32)
        out[f"blk{j}_v_b"] = b.numpy().astype(np.float32)

        # talking_heads: Conv1d(num_heads, num_heads, 1) => 融合后仍是 [heads, heads, 1]
        th_w = a.talking_heads.weight.detach()
        out[f"blk{j}_th_w"] = th_w.numpy().astype(np.float32)

        w, b = bn_fused(a.proj_conv.weight, a.proj_conv.bias, a.proj_bn)
        out[f"blk{j}_proj_w"] = w.numpy().astype(np.float32)
        out[f"blk{j}_proj_b"] = b.numpy().astype(np.float32)

        m = blk.mlp
        w, b = bn_fused(m.fc1_conv.weight, m.fc1_conv.bias, m.fc1_bn)
        out[f"blk{j}_fc1_w"] = w.numpy().astype(np.float32)
        out[f"blk{j}_fc1_b"] = b.numpy().astype(np.float32)
        w, b = bn_fused(m.fc2_conv.weight, m.fc2_conv.bias, m.fc2_bn)
        out[f"blk{j}_fc2_w"] = w.numpy().astype(np.float32)
        out[f"blk{j}_fc2_b"] = b.numpy().astype(np.float32)

    # ---------- 分类头 ----------
    out["head_w"] = model.head.weight.detach().numpy().astype(np.float32)
    out["head_b"] = model.head.bias.detach().numpy().astype(np.float32)

    # ---------- 元信息 ----------
    out["meta"] = np.array(
        [
            32,   # img_size
            16,   # patch_size
            3,    # in_channels
            10,   # num_classes
            256,  # embed_dims
            8,    # num_heads
            4,    # mlp_ratio
            2,    # depths
            4,    # T
        ],
        dtype=np.int64,
    )

    os.makedirs(os.path.join(ROOT, "rust-sdt", "artifacts"), exist_ok=True)
    path = os.path.join(ROOT, "rust-sdt", "artifacts", "sdt_reference.npz")
    np.savez(path, **out)
    print("saved:", path)
    print("logits:", logits_mean.numpy())
    print("keys:", sorted(out.keys())[:8], "... total", len(out))

    # ---------- 新增中间张量：数量与形状抽查 ----------
    inter_keys = [
        "sps_proj_lif", "sps_proj1_lif", "sps_proj2_lif", "sps_proj2_pool",
        "sps_x_feat", "sps_proj3_lif", "sps_rpe_out", "head_lif_out",
    ]
    for j in range(len(model.block)):
        inter_keys += [
            f"blk{j}_shortcut_lif", f"blk{j}_q_lif", f"blk{j}_k_lif", f"blk{j}_v_lif",
            f"blk{j}_kv_sum", f"blk{j}_kv_lif", f"blk{j}_x_attn", f"blk{j}_proj_out",
            f"blk{j}_ssa_out", f"blk{j}_mlp_fc1_lif", f"blk{j}_mlp_fc1_out",
            f"blk{j}_mlp_fc2_lif", f"blk{j}_mlp_out", f"blk{j}_mlp_res",
        ]
    print("intermediate keys:", len(inter_keys))
    for k in inter_keys:
        print(f"  {k}: {out[k].shape}")


if __name__ == "__main__":
    main()
