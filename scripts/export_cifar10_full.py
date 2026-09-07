# -*- coding: utf-8 -*-
"""
CIFAR-10 全量数据导出：与 scripts/export_cifar10.py 同口径，但导出完整训练/测试集。

导出文件: data/cifar10_full/cifar10_data.npz
数组内容:
  - train_x: f32 [50000, 3, 32, 32]，像素归一化到 [0,1]（x/255.0）
  - train_y: i64 [50000]
  - test_x:  f32 [10000, 3, 32, 32]
  - test_y:  i64 [10000]

数据来源：仓库内已解压的 data/cifar10/cifar-10-batches-py（原始 export 时已下载），
不重复下载。解析方式与 export_cifar10.py::load_via_raw 一致
（pickle latin1，b'data' 为 [N,3072] uint8 行主序 CHW）。

运行方式（PowerShell，仓库根目录）:
  D:\Anaconda\envs\Pytorch-CUDA\python.exe scripts/export_cifar10_full.py
"""

import os
import pickle
import sys

import numpy as np

BATCH_DIR = os.path.join("data", "cifar10", "cifar-10-batches-py")
OUT_DIR = os.path.join("data", "cifar10_full")
OUT_PATH = os.path.join(OUT_DIR, "cifar10_data.npz")
IMG_C, IMG_H, IMG_W = 3, 32, 32
DATA_MEAN = [0.4914, 0.4822, 0.4465]
DATA_STD = [0.2470, 0.2435, 0.2616]


def read_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d["data"], np.array(d["labels"], dtype=np.int64)


def main():
    if not os.path.isdir(BATCH_DIR):
        print("[fatal] 缺少原始批次目录: {}（先运行 export_cifar10.py）".format(BATCH_DIR))
        sys.exit(1)

    tr_parts, tr_labels = [], []
    for i in range(1, 6):
        x, y = read_batch(os.path.join(BATCH_DIR, "data_batch_{}".format(i)))
        tr_parts.append(x)
        tr_labels.append(y)
        print("  已读取 data_batch_{} ({} 样本)".format(i, x.shape[0]))
    train_data = np.concatenate(tr_parts, axis=0)          # [50000, 3072] uint8 CHW
    train_labels = np.concatenate(tr_labels, axis=0)
    te_x_raw, te_y = read_batch(os.path.join(BATCH_DIR, "test_batch"))
    print("  已读取 test_batch ({} 样本)".format(te_x_raw.shape[0]))

    tr_x = train_data.reshape(-1, IMG_C, IMG_H, IMG_W).astype(np.float32) / 255.0
    te_x = te_x_raw.reshape(-1, IMG_C, IMG_H, IMG_W).astype(np.float32) / 255.0
    tr_y = train_labels.astype(np.int64)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        OUT_PATH,
        train_x=tr_x,
        train_y=tr_y,
        test_x=te_x,
        test_y=te_y.astype(np.int64),
        data_mean=np.array(DATA_MEAN, dtype=np.float64),
        data_std=np.array(DATA_STD, dtype=np.float64),
    )
    print("[save] {}".format(OUT_PATH))
    print("train_x: {} range=[{:.4f}, {:.4f}]".format(tr_x.shape, tr_x.min(), tr_x.max()))
    print("test_x:  {} ".format(te_x.shape))
    counts = np.bincount(tr_y, minlength=10)
    print("train 每类样本数: {}".format(counts.tolist()))


if __name__ == "__main__":
    main()
