"""Run after clone and separate waveform transfer. No training or file mutation."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(2**20),b''):h.update(part)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-cpu',action='store_true');args=parser.parse_args()
    for item in json.loads((ROOT/'data_manifest.json').read_text(encoding='utf-8')):
        path=ROOT/item['path']
        if not path.is_file():raise FileNotFoundError('请另传文件到这里：'+str(path))
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('数据大小/SHA不匹配：'+str(path))
        print('数据SHA通过：'+item['path'],flush=True)
    import torch
    import numpy as np
    import run_lwc_stability
    from types import SimpleNamespace
    from run_laspi_gfsl import load_samples
    cfg=SimpleNamespace(**json.loads((ROOT/'lwc_stability_server.json').read_text(encoding='utf-8')))
    rows,_,_=load_samples(ROOT/cfg.reference_cache,metadata_only=True)
    with (ROOT/cfg.waveform_cache/'waveforms.npy').open('rb') as stream:wave=np.load(stream,allow_pickle=False)
    if wave.shape!=(len(rows),8192) or not np.isfinite(wave).all():raise ValueError('波形行数/长度/有限值错误')
    print(f'Python={sys.version.split()[0]} Torch={torch.__version__} CUDA runtime={torch.version.cuda} 可用={torch.cuda.is_available()}',flush=True)
    if torch.cuda.is_available():
        device=torch.device('cuda');probe=torch.randn(16,16,device=device);value=float((probe@probe.T).mean());torch.cuda.synchronize()
        print(f'GPU={torch.cuda.get_device_name()} 运算探针={value:.6f}',flush=True)
    elif not args.allow_cpu:raise RuntimeError('当前PyTorch未检测到CUDA；检查驱动/安装的PyTorch版本')
    subprocess.run([sys.executable,'-B',str(ROOT/'run_lwc_stability.py'),'--config',str(ROOT/'lwc_stability_server.json'),'--check'],cwd=ROOT,check=True)
    print('服务器启动前检查通过。',flush=True)


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    main()
