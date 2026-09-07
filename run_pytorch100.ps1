# PyTorch 100-epoch 基准训练（用 Pytorch-CUDA conda 环境）
# --init-npz：与 Burn 侧对齐（同一 NPZ 初始权重 + 静态校准；校准仅在此分支内执行）
$env:PYTHONIOENCODING = "utf8"
$py = "D:\Anaconda\envs\Pytorch-CUDA\python.exe"
Set-Location "F:\RustProjects\Spike-Driven-Transformer"
& $py scripts\train_pytorch_reference.py --epochs 100 --seed 42 --init-npz rust-sdt/artifacts/sdt_reference.npz 2>&1
