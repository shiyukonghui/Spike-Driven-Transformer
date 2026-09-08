"""MNIST → rust-sdt loader 格式导出（fc 模式复现 HyperscaleES 主场）。

读取 HyperscaleES/data/MNIST/raw 的 idx 文件，/255 归一化，中心补边到 32×32，
灰度复制为 3 通道，存为 train_x/train_y/test_x/test_y npz。
用法：python export_mnist.py --src <raw目录> --out <输出npz>
"""
import argparse
import gzip
import os
import struct

import numpy as np


def read_idx_images(path: str) -> np.ndarray:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"图像 idx magic 异常: {magic}"
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def read_idx_labels(path: str) -> np.ndarray:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"标签 idx magic 异常: {magic}"
        return np.frombuffer(f.read(n), dtype=np.uint8)


def to_loader_format(images: np.ndarray, labels: np.ndarray) -> dict:
    n = images.shape[0]
    x = images.astype(np.float32) / 255.0
    # 28×28 → 中心补边 32×32（offset 2）
    padded = np.zeros((n, 32, 32), dtype=np.float32)
    padded[:, 2:30, 2:30] = x
    # 灰度 → 3 通道 [N,3,32,32]
    x3 = np.repeat(padded[:, None, :, :], 3, axis=1)
    return {
        "x": x3,
        "y": labels.astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"F:\PythonProject\HyperscaleES\data\MNIST\raw")
    ap.add_argument("--out", default=r"F:\RustProjects\Spike-Driven-Transformer\rust-sdt\artifacts\mnist\cifar10_data.npz")
    args = ap.parse_args()

    tr_img = read_idx_images(os.path.join(args.src, "train-images-idx3-ubyte.gz"))
    tr_lab = read_idx_labels(os.path.join(args.src, "train-labels-idx1-ubyte.gz"))
    te_img = read_idx_images(os.path.join(args.src, "t10k-images-idx3-ubyte.gz"))
    te_lab = read_idx_labels(os.path.join(args.src, "t10k-labels-idx1-ubyte.gz"))
    print(f"MNIST raw: train {tr_img.shape}, test {te_img.shape}")

    tr = to_loader_format(tr_img, tr_lab)
    te = to_loader_format(te_img, te_lab)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        train_x=tr["x"], train_y=tr["y"], test_x=te["x"], test_y=te["y"],
    )
    print(f"已写出 {args.out}（train_x {tr['x'].shape}, test_x {te['x'].shape}）")


if __name__ == "__main__":
    main()
