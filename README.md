# LASPI LWC 四组实验：服务器迁移

此目录就是待上传 GitHub 的项目根目录。包含当前四组 LWC 完整训练流程、递归导入依赖、参数、环境版本记录、数据清单与完整波形缓存。没有旧实验结果、原始 CSV 和虚拟环境。带有 DGGN 名称的两个辅助模块需要保留，但本 LWC 入口不调用原 DGGN 仓库，因此不需要上传 `DGGN/`。只运行本说明指定的 `run_lwc_stability.py`。

## 代码与数据一起上传 GitHub

将本目录作为仓库根目录，保留隐藏文件 `.gitignore` 和 `.gitattributes`。数据已经放在本目录内，不需要回原项目找文件。波形约 163 MiB，超过 GitHub 普通 Git 的单文件限制，已配置 Git LFS；不要用网页直接拖入大文件。

首次创建仓库后，在本目录执行以下 PowerShell 命令（仓库地址换成你自己新建的空仓库地址）：
```powershell
Set-Location 'H:\keyan\github_server_lwc'
git init
git lfs install --local
git add .
git lfs ls-files
git commit -m "Add complete LWC four-way training and LASPI waveform cache"
git branch -M main
git remote add origin https://github.com/你的账号/你的仓库.git
git push -u origin main
```
`git lfs ls-files` 应显示 `prepared_dggn_raw_waveforms/waveforms.npy`，表示将以 LFS 上传。服务器安装 Git LFS 后，通过 `git clone` 获取代码，再在仓库内执行 `git lfs pull` 获取完整数据；只下载到指针文件时，预检查会报数据大小／SHA不匹配。也可以直接将此完整文件夹传服务器，不经过 GitHub。

已包含的数据文件：
```
prepared_window_zscore/samples.csv
prepared_window_zscore/records.csv
prepared_window_zscore/complete.json
prepared_dggn_raw_waveforms/complete.json
prepared_dggn_raw_waveforms/waveforms.npy   # 实际波形，Git LFS管理
```

清单中的旧 H 盘路径保留为来源记录。只要现有缓存完整，当前入口不读取这些原始 CSV 路径，不需要改成服务器路径。若要重建缓存，则需另行下载原始 LASPI 数据，并修改原始文件路径；这不在当前迁移流程内。`stft.npy`、旧 `waveform_examples.npy`、旧结果与原始 LASPI 压缩包不需要迁移。

波形包含训练、验证和测试所需窗口，划分由代码按原 CSV 和目标工况生成，所有窗口不会全部用于训练；保留验证和测试数据是为了让服务器完整训练、标定并评估。不要自行按窗口拆分或重排缓存。

## 服务器重建环境（Linux 示例）

先克隆你创建的仓库并进入目录，使用 Python 3.10 建立新环境：
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
根据服务器的 NVIDIA 驱动与 GPU，按 https://pytorch.org/get-started/locally/ 选择并安装 CUDA 版 PyTorch。这里不强制安装 Windows 的 wheel，也不假定服务器支持 CUDA 12.8。原机器实测版本记录见 `environment_reference.json`，其中 torch 为 2.11.0+cu128；如服务器换版本，需重新记录并验证，不能声称跨平台逐位一致。

然后安装其余固定版本依赖：
```bash
python -m pip install -r requirements.txt
python -m pip check
```

克隆并完成 `git lfs pull` 后，执行环境、数据 SHA 和采样检查：
```bash
python -B server_preflight.py
python -B test_lwc_stability.py
```
预检查默认要求 CUDA；只为 CPU 检查可传 `--allow-cpu`，完整训练会慢。保留 4 GB GPU 上已经通过的 batch_size=32 起步，更大的显存可后续测量后调整。跨 GPU／操作系统并不保证逐位相同结果。

## 训练

先跑与本机相同的预实验，检查服务器联通与指标：
```bash
python -u -B run_lwc_stability.py --config lwc_stability_pilot.json
```
完整实验：
```bash
python -u -B run_lwc_stability.py --config lwc_stability_server.json
```
共 2160 个适配任务，终端带进度条；服务器配置关闭弹窗，只写结果和权重。远程连接断开也要继续运行时，可使用已有作业调度器，或：
```bash
nohup python -u -B run_lwc_stability.py --config lwc_stability_server.json > training.log 2>&1 &
tail -f training.log
```
续跑时配置保持相同，替换成实际输出目录：
```bash
python -u -B run_lwc_stability.py --config lwc_stability_server.json --resume results/lwc_stability_server
python -B audit_lwc_stability.py --result results/lwc_stability_server
```
已经完成的阶段和任务会跳过；尚未完成的源阶段从该阶段起点重跑，不是恢复优化器的每一步状态。

## 取回结果和画图

把服务器结果目录整体下载到本地原工程的 `results/` 下，目录名称可以保留。用本地带桌面的环境执行：
```powershell
& 'D:\python_project\.venv\Scripts\python.exe' -B show_lwc_stability.py --result 'H:\keyan\9_23\LASPI_GFSL\results\lwc_stability_server'
```
这样显示中文窗口，不自动保存图像。无桌面服务器不会自动弹窗。方法、限制和已完成的本地预实验负结果详见 `README_LWC_STABILITY.md`。

本目录已在 Windows 中检查依赖导入、数据相对路径和采样。尚未在你的目标 Linux 服务器或其 GPU 上验证，不等同于服务器实测完成。
