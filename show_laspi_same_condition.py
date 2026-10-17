"""Chinese interactive figures; no automatic image exports."""
from pathlib import Path
import argparse,os,json,csv,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib_cache'))
import matplotlib.pyplot as plt
from scipy.signal import welch
from prepare_laspi import CN,normalize_windows


def show(result,draw_only=False):
    result=Path(result);summary=json.loads((result/'summary.json').read_text(encoding='utf-8'));config=json.loads((result/'config.json').read_text(encoding='utf-8'))
    protocol='跨工况' if config.get('protocol')=='cross_condition_702010' else '同工况'
    incomplete=not (result/'execution.json').exists()
    completed=None;total=None
    if protocol=='跨工况':
        records=json.loads((result/'metrics.json').read_text(encoding='utf-8'))
        completed=len({(r['condition'],r['seed'],r['modality'],r['method']) for r in records})
        total=len(config['cases'])*len(config['seeds'])*len(config['modalities'])*2
        incomplete=completed<total
    preview=(f'【未完成预览 {completed}/{total}组】' if completed is not None else '【未完成预览】') if incomplete else ''
    if incomplete:print(preview+'，图中的均值只代表已经完成的模型，不是最终结果。',flush=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    labels={'none':'仅去均值','window_zscore':'逐窗标准化'};heads={'deterministic':'确定性','bayesian':'贝叶斯'};mods={'vibration':'振动','multisignal':'七通道'}
    scope='（未见工况子集；全测试集另报）' if protocol=='跨工况' else ''
    fig,axes=plt.subplots(2,3,figsize=(17,9));fig.canvas.manager.set_window_title(preview+protocol+'：归一化与马氏距离对照');fig.suptitle(preview+protocol+scope+'：识别、校准与确认（所有分类均用马氏距离）')
    for modality,ax in zip(config['modalities'],axes[0,:2]):
        for norm in config['normalizations']:
            for method in ('deterministic','bayesian'):
                ss=sorted([s for s in summary if s['normalization']==norm and s['modality']==modality and s['method']==method],key=lambda s:s['k'])
                ax.plot([s['k'] for s in ss],[s['novel_f1'] for s in ss],marker='o',label=labels[norm]+'＋'+heads[method])
        ax.set(title=mods[modality]+'：复合故障F1',xlabel='每种复合类支持窗口数',ylabel='F1',ylim=(0,1));ax.legend(fontsize=8);ax.grid(alpha=.2)
    five=[s for s in summary if s['k']==5];names=[labels[s['normalization']]+'\n'+mods[s['modality']]+'／'+heads[s['method']] for s in five]
    for ax,key,title in [(axes[0,2],'macro_f1','5-shot七类宏平均F1'),(axes[1,0],'ece','5-shot校准误差ECE（越小越好）'),(axes[1,1],'coverage','5-shot确认覆盖率'),(axes[1,2],'selective_error','5-shot确认样本错误率')]:
        values=[np.nan if s[key] is None else s[key] for s in five];ax.bar(np.arange(len(names)),values);ax.set_title(title);ax.set_xticks(np.arange(len(names)),names,rotation=45,ha='right',fontsize=7);ax.set_ylim(0,1)
    fig.tight_layout(rect=(0,0,1,.94))
    # Fixed earliest complete condition, first seed, first support draw; no best-case selection.
    modality=config['modalities'][-1];norm=config['normalizations'][-1];seed=config['seeds'][0]
    bundles=sorted((result/'predictions').glob(f'{norm}_{modality}_bayesian_*_s{seed}.npz'));chosen=bundles[0]
    with np.load(chosen) as n:
        ids=np.flatnonzero(n['shots']==5);j=int(ids[0] if len(ids) else 0);p=n['probabilities'][j];y=n['true_labels'];mi=n['uncertainties'][j,:,2];accepted=n['acceptance'][j]
        cm=np.zeros((7,7));np.add.at(cm,(y,p.argmax(-1)),1);cm=cm/np.maximum(cm.sum(-1,keepdims=True),1)
    fig2,axs=plt.subplots(1,3,figsize=(16,5));fig2.canvas.manager.set_window_title(preview+protocol+'：固定示例混淆矩阵与不确定性');fig2.suptitle(preview+chosen.stem+'（固定首例，不代表全条件平均）')
    im=axs[0].imshow(cm,cmap='Blues',vmin=0,vmax=1);axs[0].set_xticks(range(7),CN,rotation=45,ha='right',fontsize=8);axs[0].set_yticks(range(7),CN,fontsize=8);axs[0].set(xlabel='预测类别',ylabel='真实类别',title='按真实类别归一化的混淆矩阵')
    for r in range(7):
        for c in range(7):axs[0].text(c,r,f'{cm[r,c]:.0%}',ha='center',va='center',fontsize=7,color='white' if cm[r,c]>.6 else 'black')
    fig2.colorbar(im,ax=axs[0],fraction=.045);correct=p.argmax(-1)==y
    for mask,label in [(correct,'预测正确'),(~correct,'预测错误')]:axs[1].hist(mi[mask],bins=25,alpha=.5,label=label,density=True)
    axs[1].set(title='特征头权重互信息MI',xlabel='MI',ylabel='密度');axs[1].legend()
    ranking=np.argsort(-p.max(-1));risk=1-np.cumsum(correct[ranking])/np.arange(1,len(y)+1);axs[2].plot(np.arange(1,len(y)+1)/len(y),risk)
    if accepted.any():axs[2].scatter(accepted.mean(),1-correct[accepted].mean(),color='red',label='当前确认规则');axs[2].legend()
    axs[2].set(title='风险—覆盖曲线（仅描述测试）',xlabel='覆盖率',ylabel='错误率',ylim=(0,1));fig2.tight_layout(rect=(0,0,1,.92))
    records=list(csv.DictReader((ROOT/'prepared'/'records.csv').open(encoding='utf-8-sig')))
    with (ROOT/'prepared'/'waveform_examples.npy').open('rb') as stream:wave=np.load(stream,allow_pickle=False)
    fig3,axs=plt.subplots(2,3,figsize=(15,7));fig3.canvas.manager.set_window_title('逐窗归一化：时域与频谱');fig3.suptitle('同一原始窗口：归一化改变幅值尺度，不改变频率位置')
    for col,c in enumerate((0,5,6)):
        index=next(i for i,r in enumerate(records) if int(r['label'])==c);x=np.asarray(wave[index:index+1,3:4],np.float64)
        centered=normalize_windows(x,'none')[0,0];normalized=normalize_windows(x,'window_zscore')[0,0]
        time=np.arange(len(centered))/12800
        axs[0,col].plot(time,normalized,label='标准化后');axs[0,col].set(title=CN[c]+'：标准化振动',xlabel='时间（秒）',ylabel='标准化幅值')
        for signal,label in [(centered,'去均值'),(normalized,'逐窗标准化')]:
            f,power=welch(signal,fs=12800,nperseg=1024);axs[1,col].semilogy(f,power.clip(1e-14),label=label)
        axs[1,col].set(xlim=(0,2000),title='功率谱密度（频率间隔12.5 Hz）',xlabel='频率（Hz）',ylabel='功率谱密度');axs[1,col].legend()
    fig3.tight_layout(rect=(0,0,1,.93))
    if draw_only:
        for figure in (fig,fig2,fig3):figure.canvas.draw()
        plt.close('all');return 3
    plt.show()


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('result',type=Path);show(ap.parse_args().result)
