"""Interactive Chinese figures; no automatic image files."""
import argparse,json
from pathlib import Path
import numpy as np
from prepare_laspi import CN

def show(root):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    rows=json.loads((root/'summary.json').read_text(encoding='utf-8'));fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    fig.canvas.manager.set_window_title('DGGN公开代码适配：LASPI结果')
    for case in sorted({r['condition'] for r in rows}):
        for classifier in ('brf','mahalanobis'):
            group=sorted([r for r in rows if r['condition']==case and r['classifier']==classifier],key=lambda r:r['k'])
            for ax,key,title in zip(axes,('macro_f1','base_f1','novel_f1'),('全部类别宏平均F1','五基类F1','两复合新类F1')):
                ax.plot([r['k'] for r in group],[r[key] for r in group],'o-',label=case+' '+('平衡随机森林' if classifier=='brf' else '马氏距离'));ax.set_title(title);ax.set_xlabel('每个新类支持窗口数');ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    control_path=root/'controls.json'
    controls=json.loads(control_path.read_text(encoding='utf-8')) if control_path.exists() else None
    if controls:
        group=sorted([v for v in controls['summary'] if v['method']=='frozen_dggn_ca'],key=lambda v:v['k'])
        for ax,key in zip(axes,('macro_f1','base_f1','novel_f1')):ax.plot([v['k'] for v in group],[v[key] for v in group],'s--',label='冻结DGGN＋马氏距离')
        axes[0].legend(fontsize=7)
    for folder in sorted(root.glob('*_s*')):
        pred=folder/'predictions_5shot.npz'
        if not pred.exists():continue
        with np.load(pred) as data:truth=data['true_labels'];probs=[data['brf_probabilities'],data['mahalanobis_probabilities']]
        matrices=[np.bincount(7*truth+p.argmax(1),minlength=49).reshape(7,7) for p in probs];names=['适配＋平衡随机森林','适配＋马氏距离']
        if controls:
            matching=[v for v in controls['records'] if v['method']=='frozen_dggn_ca' and v['k']==5 and v['episode']==0 and f"{v['condition']}_s{v['seed']}"==folder.name]
            if matching:matrices.append(np.array(matching[0]['metrics']['confusion']));names.append('冻结表征＋马氏距离')
        fig,axes=plt.subplots(1,len(matrices),figsize=(6*len(matrices),5),layout='constrained');fig.canvas.manager.set_window_title('DGGN首个5-shot任务：'+folder.name)
        for ax,cm,name in zip(axes,matrices,names):
            cm=cm/np.maximum(cm.sum(1,keepdims=True),1)
            im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1);fig.colorbar(im,ax=ax);ax.set_xticks(range(7),CN,rotation=45,ha='right');ax.set_yticks(range(7),CN);ax.set_title(name+'（首个支持任务示例）');ax.set_xlabel('预测');ax.set_ylabel('真实')
            for i in range(7):
                for j in range(7):ax.text(j,i,f'{cm[i,j]:.0%}',ha='center',va='center',color='white' if cm[i,j]>.5 else 'black',fontsize=8)
    plt.show()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);args=parser.parse_args();show(args.result)
