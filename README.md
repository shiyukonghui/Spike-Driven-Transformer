# Spike-Driven Transformer ([NeurIPS2023](https://openreview.net/forum?id=9FmolyOHi5))

[Man Yao](https://scholar.google.com/citations?user=eE4vvp0AAAAJ), [Jiakui Hu](https://github.com/jkhu29), [Zhaokun Zhou](https://github.com/ZK-Zhou), [Li Yuan](https://yuanli2333.github.io/), [Yonghong Tian](https://scholar.google.com/citations?user=fn6hJx0AAAAJ), [Bo Xu](), [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&)

BICLab, Institute of Automation, Chinese Academy of Sciences

---

:rocket:  :rocket:  :rocket: **News**:

- **Jul. 04, 2023**: Release the code for training and testing.
- **Sep. 22, 2023**: Accepted as poster in NeurIPS2023.
- **Sep. 30, 2023**: Release the configs and pre-trained parameters on IN1K.
- **Feb. 15. 2024**: The [Spike-Driven Transformer V2](https://github.com/BICLab/Spike-Driven-Transformer-V2), which achieves 80.0% acc on IN1K, is now available.

## Abstract

Spiking Neural Networks (SNNs) provide an energy-efficient deep learning option due to their unique spike-based event-driven (i.e., spike-driven) paradigm. In this paper, we incorporate the spike-driven paradigm into Transformer by the proposed Spike-driven Transformer with four unique properties: i) **Event-driven**, no calculation is triggered when the input of Transformer is zero; ii) **Binary spike communication**, all matrix multiplications associated with the spike matrix can be transformed into sparse additions; iii) **Self-attention with linear complexity at both token and channel dimensions**; iv) The operations between spike-form Query, Key, and Value are mask and addition. Together, **there are only sparse addition operations** in the Spike-driven Transformer. To this end, we design a novel Spike-Driven Self-Attention (SDSA), which exploits only mask and addition operations without any multiplication, and thus having up to **87.2× lower** computation energy than vanilla self-attention. Especially in SDSA, the matrix multiplication between Query, Key, and Value is designed as the mask operation. In addition, we rearrange all residual connections in the vanilla Transformer before the activation functions to ensure that all neurons transmit binary spike signals. It is shown that the Spike-driven Transformer can achieve **77.1% top-1** accuracy on ImageNet-1K, which is the state-of-the-art result in the SNN field.

![SDSA](./imgs/Fig_1_main_idea.png)

## Requirements

```python3
timm == 0.6.12
1.10.0 <= pytorch < 2.0.0
cupy
spikingjelly == 0.0.0.0.12
tensorboard
```

> !!! Please install the spikingjelly and tensorboard correctly before raising issues about requirements. !!!

## Results on Imagenet-1K

|        **model**         | **T** | **layers** | **channels** | **Top-1 Acc** | **Power(mj)** | **Models** |
| :----------------------: | :---: | :--------: | :----------: | :-----------: | :-----------: | :--------: |
| Spike-Driven Transformer |   4   |     8      |     384      |   **72.28**   |   **3.90**    |    [link](https://drive.google.com/file/d/10oH_zkwB4FDtFLgmZ_lI8e0tFjRzrXyD/view?usp=sharing)    |
| Spike-Driven Transformer |   4   |     6      |     512      |   **74.11**   |   **3.56**    |    [link](https://drive.google.com/file/d/1hsShpFBKYpMK2TmpuoyBFORcLAuMHrx7/view?usp=sharing)    |
| Spike-Driven Transformer |   4   |     8      |     512      |   **74.57**   |   **4.50**    |    [link](https://drive.google.com/file/d/1n59WNSBgP2VyAW2nfJX2Wvx5rgNJEMXI/view?usp=sharing)    |
| Spike-Driven Transformer |   4   |     10     |     512      |   **74.66**   |   **5.53**    |    [link](https://drive.google.com/file/d/1l-c3QY5r4IFmYUmGPZXRHP_W7iC1sdP8/view?usp=sharing)    |
| Spike-Driven Transformer |   4   |     8      |     768      |   **77.07**   |   **6.09**    |    [link](https://drive.google.com/file/d/1R-MaeFV8d2Y0pIGBSjklOGWhaF8dLHf4/view?usp=sharing)    |

## Train & Test

![The architecture of Spike-Driven-Transformer.](./imgs/Fig_2_network_architecture.png)

The hyper-parameters are in `./conf/`.


Train:

```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 train.py -c /the/path/of/conf --model sdt --spike-mode lif
```

Test:

```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 firing_num.py -c /the/path/of/conf --model sdt --spike-mode lif --resume /the/path/of/parameters --no-resume-opt

# for 288 x 288 resolution
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 firing_num.py -c /the/path/of/conf --model sdt --spike-mode lif --resume /the/path/of/parameters --no-resume-opt --large-valid
```

Result and explainability:

![The Attention Map of Spike-Driven Transformer in ImageNet.](./imgs/Fig_3_attention_map.png)

## Rust + Burn 迁移（rust-sdt/）

将本项目的 SDT 模型迁移到 Rust + Burn 0.21.0（wgpu GPU 后端），并与 PyTorch 基准做训练结果对照，
验收标准（2026-09-08 更新）：100 epoch 训练后，最终 val top-1 相对差 ≤ 5%（分母=pytorch 值）
且 train loss 绝对差 ≤ 0.1 判 PASS。当前状态：**PASS**（top1 相对差 0.15%、loss 绝对差 0.031）。

### 目录结构

```
├── rust-sdt/                 # Rust 实现（Burn）
│   ├── src/                  # model/ops/train/loader/check/config/tensor_io
│   ├── artifacts/            # 权重、数据、报告、CSV（sdt_reference.npz / cifar10_data.npz / forward_report.txt / train_report.txt / *.csv）
│   └── probe_vram.ps1        # VRAM 轮询探测脚本
├── scripts/
│   ├── export_reference.py       # 导出 PyTorch 权重与中间张量
│   ├── export_cifar10.py         # 导出 CIFAR-10 子集 npz
│   └── train_pytorch_reference.py # PyTorch 小规模训练基准
└── .trae/specs/migrate-sdt-to-burn/ # 迁移 spec / tasks / checklist
```

### 使用命令（在 rust-sdt/ 下）

```shell
# 1) 前向逐模块对照（rel_l2<5% 且 max_abs<0.05 判 PASS）
cargo run --release -- forward-check

# 2) Burn 侧小规模训练（T=4 全程可跑：校准 + 训练 + eval 零 OOM，~21s/epoch）
cargo run --release -- train --epochs 2

# 3) 训练结果对照（读取 artifacts/train_burn.csv 与 artifacts/train_pytorch.csv）
cargo run --release -- train-compare
```

### 当前状态（2026-09-08）

| 项目 | 状态 |
| --- | --- |
| 构建（workspace + burn 0.21.0） | 完成 |
| forward-check 前向对照 | PASS（logits rel_l2=0，与 PyTorch 逐 bit 一致） |
| PyTorch 基准（train_pytorch.csv） | epoch2: loss 1.8307 / top1 33.95% |
| burn 0.21 显存问题 | **已修复**（T=4 全程零 OOM，~21s/epoch，0.18 基线 79s/epoch 的 3.8×） |
| Burn 训练（0.21 + 显存修复） | 100 epoch：ep100 loss 1.8257→0.1289 / top1 65.55%，全程零 OOM |
| train-compare（100 epoch 对照） | **PASS**（top1 相对差 0.15% ≤5%；loss 绝对差 0.031 ≤0.1；口径见 check_log 七节） |
| EggRoll-ES 演化策略训练（无反传） | 完成：`train-es` 双模式——factored 因式噪声前向（逐图候选，**每次更新 8000 个独立噪声方向**，SPS 冻结，4.4GB）best_val **16.65%**；cache ΔW 物化（800 方向）15.70%。SGD 基线 65.55%；ES 收敛快 2 倍但受 100 次更新/SPS 冻结制约，详见 check_log 八/九节 |
| TSES 温度松弛 ES（数学指导优化） | 完成：按 ES_MANIFOLD_NOTE.md 定理 3 实现 σ(β(h−thr)) 松弛 LIF fitness 前向 + β 退火 4→16；val **17.10%**（超硬脉冲 v2 的 16.65%），且**终点=峰值、尾部单调上升**（v2 峰值后衰减）——"信号通道恢复"的机制预测在全尺度复现；数值验证（引理 1 平板流形、一致性 |cos|→0.99）见 `rust-sdt/ES_MANIFOLD_NOTE.md` §9 与 `scripts/verify_es_math.py` |
| 全量 CIFAR-10 ES 测试（50000/10000） | 已终止（用户裁决：34 epoch，best_val **17.68%**）。SGD 参考 5 epoch 即 **58.95%**（655s）——方向数 ×6.25 仅 +1.0pp，确认 ES 与反传的差距是数量级且不随数据缩小，ES 定位为无梯度场景工具而非主优化器，详见 check_log 第十节 |

显存修复摘要（三叠加根因，详见 [rust-sdt/check_log.txt](rust-sdt/check_log.txt)）：
1. burn-fusion 0.21 延迟 drop（ContinueDrop 不触发 drain）→ 本地补丁恢复 0.18 语义
   （`patches/burn-fusion`，`BURN_FUSION_CONTINUE_DROP_DRAIN=0` 可关）；
2. 纯前向路径（eval/校准）autodiff 图节点不释放 → eval 改走内层 wgpu 后端 +
   校准/eval 逐批 sync+cleanup；
3. cubecl SlicedPages 池不自动回收 → 保留阶段边界清理。
诊断设施默认零开销，环境变量门控（`BURN_FUSION_LOG`、`BURN_SDT_POOL_DIAG`、
`SDT_PROBE_*`、`SDT_BATCH_SYNC`、`SDT_BATCH_CLEANUP`）。

## Data Prepare

- use `PyTorch` to load the CIFAR10 and CIFAR100 dataset.
- use `SpikingJelly` to prepare and load the Gesture and CIFAR10-DVS dataset.

Tree in `./data/`.

```shell
.
├── cifar-100-python
├── cifar-10-batches-py
├── cifar10-dvs
│   ├── download
│   ├── events_np
│   ├── extract
│   ├── frames_number_10_split_by_number
│   └── frames_number_16_split_by_number
├── cifar10-dvs-tet
│   ├── test
│   └── train
└── DVSGesturedataset
    ├── download
    ├── events_np
    │   ├── test
    │   └── train
    ├── extract
    │   └── DvsGesture
    ├── frames_number_10_split_by_number
    │   ├── download
    │   ├── test
    │   └── train
    └── frames_number_16_split_by_number
        ├── test
        └── train
```

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Contact Information

```
@inproceedings{yao2023spikedriven,
title={Spike-driven Transformer},
author={Man Yao and JiaKui Hu and Zhaokun Zhou and Li Yuan and Yonghong Tian and Bo XU and Guoqi Li},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=9FmolyOHi5}
}
```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `manyao@ia.ac.cn` and `jkhu29@stu.pku.edu.cn`.
