"""Four-way Chinese interactive dashboard; images are never auto-saved."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from prepare_laspi import CN

NAMES={'single':'① LWC单判别分支','parallel':'② 稳定分支无交流','frozen_exchange':'③ 冻结稳定参考＋交流','trainable_exchange':'④ 稳定参考参与更新'}

def show(root):
    import matplotlib.pyplot as plt
    import torch
    from lwc_stability_model import DualLWC
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'));cfg=SimpleNamespace(**json.loads((root/'config.json').read_text(encoding='utf-8')))
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');fig.canvas.manager.set_window_title('LWC稳定信息：四组配对消融')
    for mode,name in NAMES.items():
        rows=sorted([r for r in summary if r['mode']==mode],key=lambda r:r['k'])
        if not rows:continue
        for ax,key,title in zip(axes.flat,('macro_f1','base_f1','novel_f1','inner_component_f1','outer_component_f1','stable_change'),('七类宏平均F1','五基类F1','复合新类F1','内圈成分F1','外圈成分F1','稳定分支参数最大变化')):
            ax.plot([r['k'] for r in rows],[r[key] for r in rows],'o-',label=name);ax.set_title(title);ax.set_xlabel('每个新类支持窗口数');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    folders=sorted(p for p in root.glob('*_s*') if (p/'common_pretrain').exists())
    for folder in folders[:1]:
        fig,axes=plt.subplots(2,2,figsize=(13,11),layout='constrained');fig.canvas.manager.set_window_title('四组首个5-shot任务混淆矩阵')
        for ax,(mode,name) in zip(axes.flat,NAMES.items()):
            path=folder/mode/'predictions_5shot.npz'
            if not path.exists():ax.set_visible(False);continue
            with np.load(path) as data:p=data['raw_probabilities'];truth=data['true_labels']
            cm=np.bincount(7*truth+p.argmax(1),minlength=49).reshape(7,7);cm=cm/np.maximum(cm.sum(1,keepdims=True),1)
            im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=ax);ax.set_xticks(range(7),CN,rotation=45,ha='right');ax.set_yticks(range(7),CN);ax.set_title(name+'（示例）',fontsize=10)
            for i in range(7):
                for j in range(7):ax.text(j,i,f'{cm[i,j]:.0%}',ha='center',va='center',color='white' if cm[i,j]>.5 else 'black',fontsize=8)
        model=DualLWC(cfg,'frozen_exchange');checkpoint=folder/'source_frozen_exchange'/'best.pt'
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained');fig.canvas.manager.set_window_title('LWC滤波器：判别与稳定分支')
            for ax,encoder,label in zip(axes,(model.discriminative,model.stable),('判别分支','稳定分支')):
                kernels=encoder.wavelet.kernels().detach().numpy()[:,0];response=np.abs(np.fft.rfft(kernels,n=4096));frequency=np.fft.rfftfreq(4096,1/25600)
                for i in np.linspace(0,len(kernels)-1,5).astype(int):
                    r=response[i]/max(response[i].max(),1e-12);peak=frequency[int(r.argmax())];ax.plot(frequency,r,label=f'核{i+1} 峰值{peak:.0f}Hz')
                ax.set_xlim(0,12800);ax.set_xlabel('频率（Hz）');ax.set_ylabel('归一化频率响应');ax.set_title(label+'：实际核频响');ax.legend(fontsize=7)
            from run_laspi_gfsl import load_samples
            from run_dggn_laspi import prepare_waveforms
            rows,y,_=load_samples(root.parent.parent/cfg.reference_cache,metadata_only=True);wave=prepare_waveforms(cfg,rows)
            with (folder/'single'/'predictions_5shot.npz').open('rb') as stream:
                with np.load(stream) as data:indices=data['query_indices'];truth=data['true_labels']
            chosen=[int(indices[np.flatnonzero(truth==c)[0]]) for c in (0,3,4,5)]
            with torch.no_grad():sequence,images=model.stable.wavelet(torch.tensor(wave[chosen].astype(np.float32)))
            fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');fig.canvas.manager.set_window_title('稳定分支LWC尺度—时间响应图')
            for ax,image,c in zip(axes.flat,images.numpy(),(0,3,4,5)):
                im=ax.imshow(image,origin='lower',aspect='auto',extent=[0,.32,0,cfg.wavelets],cmap='magma');fig.colorbar(im,ax=ax,label='log(1＋局部响应能量)');ax.set_xlabel('时间（秒）');ax.set_ylabel('小波核编号（不是线性Hz轴）');ax.set_title(CN[c])
    plt.show()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);args=parser.parse_args();show(args.result)
