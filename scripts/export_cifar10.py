# -*- coding: utf-8 -*-
"""
CIFAR-10 数据导出脚本：将 CIFAR-10 图像导出为 NPZ，供 Rust 端 (rust-sdt) 读取。

导出文件: rust-sdt/artifacts/cifar10_data.npz
数组内容:
  - train_x: f32 [8000, 3, 32, 32]，像素归一化到 [0,1]（x/255.0）
  - train_y: i64 [8000]，对应标签
  - test_x:  f32 [2000, 3, 32, 32]
  - test_y:  i64 [2000]
  - data_mean: f64 [3]（仅元信息，本任务训练不使用）
  - data_std:  f64 [3]（仅元信息，本任务训练不使用）

子集选择规则（与 PyTorch 侧对照脚本保持一致）:
  选中索引 = np.random.RandomState(SEED).permutation(len(dataset))[:N]
  即：用固定种子 42 生成长度的随机排列，取前 N 个索引。
  train/test 各自独立执行一次该规则（train 用 train 集，test 用 test 集）。

数据获取策略:
  1. 优先使用 torchvision.datasets.CIFAR10(download=True) 下载到 ./data/cifar10；
  2. 若 torchvision 缺失或下载失败，则用 urllib 直接下载
     https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 并解压，
     手动解析 data_batch_* 与 test_batch（pickle 序列化的 dict：
     b'data' 为 [N, 3072] uint8 行主序 CHW，b'labels' 为标签列表）。

运行方式（PowerShell）:
  D:\Anaconda\envs\Pytorch-CUDA\python.exe scripts/export_cifar10.py
"""

import os
import pickle
import sys
import tarfile
import urllib.request

import numpy as np

# ---------------------------------------------------------------------------
# 导出常量（供 PyTorch 侧对照脚本复用，保证子集选择规则一致）
# ---------------------------------------------------------------------------
SEED = 42          # 固定随机种子
N_TRAIN = 8000     # 训练子集大小
N_TEST = 2000      # 测试子集大小
NUM_CLASSES = 10   # 类别数

# CIFAR-10 常量
IMG_C, IMG_H, IMG_W = 3, 32, 32
# CIFAR 原始数据每行 3072 = 3*32*32，行主序 CHW

# 数据集根目录（相对仓库根目录运行）与输出路径
DATA_ROOT = os.path.join("data", "cifar10")
OUT_PATH = os.path.join("rust-sdt", "artifacts", "cifar10_data.npz")

# 官方下载地址（回退方案）
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# 下载超时（秒），避免无网时长时间阻塞
DOWNLOAD_TIMEOUT = 120

# 常用的 CIFAR-10 通道均值/方差（仅作为元信息导出，训练保持 [0,1] 不做标准化）
DATA_MEAN = [0.4914, 0.4822, 0.4465]
DATA_STD = [0.2470, 0.2435, 0.2616]


def _select_indices(n_total, n_select, seed=SEED):
    """按约定规则生成选中索引：RandomState(seed).permutation(n_total)[:n_select]。

    说明：与后续 PyTorch 侧对照脚本使用同一规则，保证两边子集完全一致。
    """
    return np.random.RandomState(seed).permutation(n_total)[:n_select]


# ---------------------------------------------------------------------------
# 数据获取：优先 torchvision，失败后回退到手动下载解析
# ---------------------------------------------------------------------------

def load_via_torchvision():
    """使用 torchvision 加载 CIFAR-10（download=True）。

    返回 (train_data, train_labels, test_data, test_labels) 或 None（不可用时）。
    train_data/test_data 为 uint8 [N, 3, 32, 32]（torchvision 直接给 CHW）。
    """
    try:
        from torchvision.datasets import CIFAR10  # noqa: 延迟导入，容错
    except ImportError:
        print("[fallback] torchvision 不可用，改用直接下载方式")
        return None
    try:
        # torchvision 会在目标目录生成 cifar-10-batches-py
        train_set = CIFAR10(root=DATA_ROOT, train=True, download=True)
        test_set = CIFAR10(root=DATA_ROOT, train=False, download=True)
    except Exception as e:  # noqa: 网络错误/超时等一律回退
        print("[fallback] torchvision 下载失败: {}".format(e))
        return None
    # train_set.data: uint8 ndarray [50000, 32, 32, 3]（torchvision 内部为 HWC）
    # 这里直接转回 CHW uint8，避免经过 PIL
    tr = np.ascontiguousarray(train_set.data.transpose(0, 3, 1, 2))
    te = np.ascontiguousarray(test_set.data.transpose(0, 3, 1, 2))
    tr_y = np.array(train_set.targets, dtype=np.int64)
    te_y = np.array(test_set.targets, dtype=np.int64)
    return tr, tr_y, te, te_y


def _download_tar(dest_path):
    """用 urllib 直接下载 CIFAR-10 tar.gz（带超时，进度打印）。"""
    print("[download] {}".format(URL))
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, \
            open(dest_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 20)  # 每次 1 MiB
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print("\r  {:.1f} / {:.1f} MiB".format(got / 1048576, total / 1048576), end="")
        print()
    print("[download] 完成: {}".format(dest_path))


def _extract_tar(dest_path):
    """解压 tar.gz 到 DATA_ROOT。"""
    print("[extract] {}".format(dest_path))
    with tarfile.open(dest_path, "r:gz") as tar:
        tar.extractall(DATA_ROOT)


def _read_batch(path):
    """解析单个 CIFAR-10 python 批次文件（pickle 序列化的 dict）。"""
    # Python 3 需要用 encoding='latin1'（或 'bytes'）读取 py2 序列化数据
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    data = d["data"]  # uint8 [N, 3072] 行主序 CHW
    labels = np.array(d["labels"], dtype=np.int64)  # [N]
    return data, labels


def load_via_raw():
    """回退方案：直接下载 tar.gz 并手动解析 data_batch_* 与 test_batch。

    返回 (train_data, train_labels, test_data, test_labels)，
    data 均为 uint8 [N, 3072]（保持原始行主序，后续 reshape(3,32,32)）。
    """
    os.makedirs(DATA_ROOT, exist_ok=True)
    tar_path = os.path.join(DATA_ROOT, "cifar-10-python.tar.gz")
    # 已存在 tar 包则跳过下载；否则重新下载（覆盖不完整文件）
    if not os.path.exists(tar_path):
        try:
            _download_tar(tar_path)
        except Exception as e:
            print("[error] 下载失败: {}".format(e))
            return None
    if not os.path.exists(os.path.join(DATA_ROOT, "cifar-10-batches-py", "test_batch")):
        try:
            _extract_tar(tar_path)
        except Exception as e:
            print("[error] 解压失败: {}".format(e))
            return None
    batch_dir = os.path.join(DATA_ROOT, "cifar-10-batches-py")
    # 训练集 5 个批次
    tr_parts, tr_labels = [], []
    for i in range(1, 6):
        path = os.path.join(batch_dir, "data_batch_{}".format(i))
        if not os.path.exists(path):
            print("[error] 缺少批次文件: {}".format(path))
            return None
        x, y = _read_batch(path)
        tr_parts.append(x)
        tr_labels.append(y)
        print("  已读取 {} ({} 样本)".format(os.path.basename(path), x.shape[0]))
    train_data = np.concatenate(tr_parts, axis=0)
    train_labels = np.concatenate(tr_labels, axis=0)
    # 测试集单个批次
    te_x, te_y = _read_batch(os.path.join(batch_dir, "test_batch"))
    print("  已读取 test_batch ({} 样本)".format(te_x.shape[0]))
    return train_data, train_labels, te_x, te_y


# ---------------------------------------------------------------------------
# 导出主流程
# ---------------------------------------------------------------------------

def export():
    """加载数据 -> 选子集 -> 归一化 -> 保存 NPZ -> 打印验证。"""
    # 1. 获取原始数据（先 torchvision，失败回退手动下载）
    raw = load_via_torchvision()
    if raw is None:
        raw = load_via_raw()
    if raw is None:
        print("[fatal] 无法获取 CIFAR-10 数据（torchvision 与直接下载均失败）")
        sys.exit(1)
    tr_data, tr_labels, te_data, te_labels = raw
    n_tr, n_te = tr_data.shape[0], te_data.shape[0]
    print("[data] 训练集 {} 样本, 测试集 {} 样本".format(n_tr, n_te))

    # 2. 按固定规则选择子集索引（保证与 PyTorch 对照脚本可复现）
    tr_idx = _select_indices(n_tr, min(N_TRAIN, n_tr))
    te_idx = _select_indices(n_te, min(N_TEST, n_te))

    # 3. 切片并整形为 [N, 3, 32, 32]
    #    原始 data_batch 是 [N, 3072] 行主序 CHW，直接 reshape(N,3,32,32)，不要转 HWC
    if tr_data.ndim == 2:  # 手动解析路径：[N, 3072]
        tr_x = tr_data[tr_idx].reshape(-1, IMG_C, IMG_H, IMG_W)
        te_x = te_data[te_idx].reshape(-1, IMG_C, IMG_H, IMG_W)
    else:  # torchvision 路径：已是 [N, 3, 32, 32]
        tr_x = tr_data[tr_idx]
        te_x = te_data[te_idx]
    tr_y = tr_labels[tr_idx].astype(np.int64)
    te_y = te_labels[te_idx].astype(np.int64)

    # 4. 归一化到 [0,1]（不做均值/方差标准化）
    tr_x = tr_x.astype(np.float32) / 255.0
    te_x = te_x.astype(np.float32) / 255.0

    # 5. 写出 NPZ（标签 i64；均值方差作为元信息导出）
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        train_x=tr_x,
        train_y=tr_y,
        test_x=te_x,
        test_y=te_y,
        data_mean=np.array(DATA_MEAN, dtype=np.float64),
        data_std=np.array(DATA_STD, dtype=np.float64),
    )
    print("[save] {}".format(OUT_PATH))

    # 6. 打印形状与基本统计验证
    print("=== 形状验证 ===")
    print("train_x: {} dtype={} range=[{:.4f}, {:.4f}]".format(
        tr_x.shape, tr_x.dtype, tr_x.min(), tr_x.max()))
    print("train_y: {} dtype={} range=[{}, {}]".format(
        tr_y.shape, tr_y.dtype, tr_y.min(), tr_y.max()))
    print("test_x:  {} dtype={} range=[{:.4f}, {:.4f}]".format(
        te_x.shape, te_x.dtype, te_x.min(), te_x.max()))
    print("test_y:  {} dtype={} range=[{}, {}]".format(
        te_y.shape, te_y.dtype, te_y.min(), te_y.max()))
    print("data_mean: {} data_std: {}".format(DATA_MEAN, DATA_STD))
    # 每类样本数粗校验（应为近似均匀的 10 类）
    counts = np.bincount(tr_y, minlength=NUM_CLASSES)
    print("train 子集每类样本数: {}".format(counts.tolist()))


if __name__ == "__main__":
    export()
