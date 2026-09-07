# VRAM 探测脚本：轮询显存占用，同时运行训练程序，用于定位显存爆炸时间点
$pollFile = "vram_poll.txt"
$exe = "F:\RustProjects\Spike-Driven-Transformer\target\release\rust-sdt.exe"
"t_sec mem_used_mib" | Out-File $pollFile

# 启动后台轮询任务（每 0.5 秒采样一次）
$pollJob = Start-Job -ScriptBlock {
    param($file)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        $m = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        "$([int]$sw.Elapsed.TotalSeconds) $m" | Add-Content $file
        Start-Sleep -Milliseconds 500
    }
} -ArgumentList (Join-Path (Get-Location) $pollFile)

# 运行训练（1 epoch，观察崩溃前的显存曲线）
$sw = [Diagnostics.Stopwatch]::StartNew()
& $exe train --epochs 1 *> train_run13.txt
"EXIT_CODE=$LASTEXITCODE t=$([int]$sw.Elapsed.TotalSeconds)s" | Add-Content train_run13.txt

# 停止轮询
Stop-Job $pollJob
Remove-Job $pollJob -Force
"probe done"
