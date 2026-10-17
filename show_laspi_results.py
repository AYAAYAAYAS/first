"""Chinese interactive LASPI dashboards. No automatic image saving."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'matplotlib_cache'))
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

NAMES={'encoder_mahalanobis':'原始CNN马氏','euclidean':'确定性欧氏','mahalanobis':'确定性马氏',
    'bayesian_mahalanobis':'贝叶斯马氏','conditioned_mahalanobis':'工况修正＋确定性','bayesian_conditioned_mahalanobis':'工况修正＋贝叶斯'}
ORDER=('encoder_mahalanobis','euclidean','mahalanobis','bayesian_mahalanobis','conditioned_mahalanobis','bayesian_conditioned_mahalanobis')


def show_results(reference,block=True):
    reference=Path(reference);config=json.loads((reference/'config.json').read_text(encoding='utf-8'))
    if not (reference/'analysis.json').exists():raise FileNotFoundError('先运行analyze_laspi.py生成analysis.json')
    analysis=json.loads((reference/'analysis.json').read_text(encoding='utf-8'))
    examples=json.loads((reference/'dashboard_display.json').read_text(encoding='utf-8'))
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    labels=config['classes_cn'];summary=analysis['summary'];figures=[]
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True);figures.append(fig)
    fig.canvas.manager.set_window_title('LASPI：跨工况GFSL与复合故障识别')
    fig.suptitle(config['title']+'\n均值先平均支持抽样；误差棒为任务/种子波动，不是轴承总体置信区间',fontsize=12)
    for modality,ax in zip(('vibration','multisignal'),axes[0,:2]):
        for j,method in enumerate(ORDER):
            ks=[k for k in config['shots'] if f'{modality}_{method}_k{k}' in summary]
            if not ks:continue
            ax.errorbar(ks,[summary[f'{modality}_{method}_k{k}']['novel_f1']['mean'] for k in ks],
                yerr=[summary[f'{modality}_{method}_k{k}']['novel_f1']['std'] for k in ks],marker='o',capsize=2,label=NAMES[method],color=f'C{j}')
        ax.set(title=('振动输入' if modality=='vibration' else '七通道输入')+'：复合类F1',xlabel='每种复合故障的支持窗数 K',ylabel='F1',ylim=(0,1))
        ax.set_xticks(config['shots']);ax.legend(fontsize=7)
    keys=[m for m in ORDER if f'multisignal_{m}_k5' in summary]
    pos=np.arange(len(keys));raw=[summary[f'multisignal_{m}_k5']['raw_ece']['mean'] for m in keys]
    calibrated=[summary[f'multisignal_{m}_k5']['ece']['mean'] for m in keys]
    axes[0,2].bar(pos-.18,raw,.36,label='未校准');axes[0,2].bar(pos+.18,calibrated,.36,label='源验证温度校准')
    axes[0,2].set_xticks(pos,[NAMES[m] for m in keys],rotation=30,fontsize=7)
    axes[0,2].set(title='5-shot：七通道目标工况ECE',ylabel='ECE（低为好）');axes[0,2].legend(fontsize=8)
    chosen=examples.get('multisignal_bayesian_conditioned_mahalanobis') or examples.get('multisignal_bayesian_mahalanobis')
    if chosen:
        cm=np.array(chosen['calibrated']['confusion']);norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        axes[1,0].imshow(norm,vmin=0,vmax=1,cmap='Blues')
        axes[1,0].set_xticks(np.arange(7),labels,rotation=45,fontsize=7);axes[1,0].set_yticks(np.arange(7),labels,fontsize=7)
        axes[1,0].set(title='首任务/首种子/首5-shot组示例',xlabel='预测类别',ylabel='真实类别')
        for i in range(7):
            for j in range(7):axes[1,0].text(j,i,f'{100*norm[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if norm[i,j]>.5 else 'black')
    scenarios=config['scenarios'];matrix=np.full((len(scenarios),len(keys)),np.nan)
    for i,scenario in enumerate(scenarios):
        for j,method in enumerate(keys):
            key=f'{scenario}_multisignal_{method}_k5'
            if key in analysis['by_scenario']:matrix[i,j]=analysis['by_scenario'][key]['novel_f1']['mean']
    axes[1,1].imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    axes[1,1].set_xticks(np.arange(len(keys)),[NAMES[m] for m in keys],rotation=35,fontsize=7)
    axes[1,1].set_yticks(np.arange(len(scenarios)),[s.replace('speed_','留转速：').replace('load_','留载荷：') for s in scenarios],fontsize=8)
    axes[1,1].set(title='七项迁移任务：5-shot复合F1')
    for i in range(len(scenarios)):
        for j in range(len(keys)):
            axes[1,1].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white')
    axes[1,2].axis('off');axes[1,2].text(0,.98,'实验定义\n\n• 五个健康/单故障基类；两种复合新类。\n\n• 每种复合各K窗，总预算2K。\n\n• 支持来自源工况，查询来自留出工况。\n\n• 原始重复记录的10份副本全部排除。\n\n• 工况修正为探索性对照，使用运行设定。\n\n• 第一篇不含载流机理损失，也不估计寿命。',va='top',fontsize=10)
    fig2,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True);figures.append(fig2)
    fig2.canvas.manager.set_window_title('LASPI：贝叶斯校准与拒判')
    fig2.suptitle('校准、拒判和未注册复合检测须分别判断；低错误率不能脱离覆盖率解释')
    if chosen:
        p=np.array(chosen['probabilities']);truth=[]
        # Resolve labels using prediction indices into the immutable cache manifest.
        sample_rows=list(csv.DictReader((Path(config['cache'])/'samples.csv').open(encoding='utf-8-sig')))
        truth=np.array([int(sample_rows[i]['label']) for i in chosen['query_indices']]);correct=p.argmax(1)==truth;conf=p.max(1)
        centers=[];observed=[]
        for j in range(10):
            use=(conf>=j/10)&((conf<(j+1)/10) if j<9 else (conf<=1))
            if use.any():centers.append(conf[use].mean());observed.append(correct[use].mean())
        ax[0,0].plot([0,1],[0,1],'k--');ax[0,0].plot(centers,observed,'o-')
        ax[0,0].set(title='可靠性图：固定示例',xlabel='预测置信度',ylabel='实际正确比例',xlim=(0,1),ylim=(0,1))
        mi=np.array(chosen['mi']);entropy=np.array(chosen['entropy'])
        ax[0,1].scatter(entropy[correct],mi[correct],s=10,alpha=.4,label='正确')
        ax[0,1].scatter(entropy[~correct],mi[~correct],s=10,alpha=.4,label='错误')
        ax[0,1].set(title='预测熵与权重互信息',xlabel='预测熵',ylabel='权重MI');ax[0,1].legend(fontsize=8)
    for j,method in enumerate(keys):
        case=examples.get(f'multisignal_{method}')
        if case:ax[0,2].plot(case['risk_coverage']['coverage'],case['risk_coverage']['risk'],label=NAMES[method],color=f'C{j}')
    ax[0,2].set(title='测试集描述性风险—覆盖曲线',xlabel='按置信度接受的比例',ylabel='接受部分错误率');ax[0,2].legend(fontsize=7)
    coverage=[summary[f'multisignal_{m}_k5']['coverage']['mean'] for m in keys]
    novel_cov=[summary[f'multisignal_{m}_k5']['novel_coverage']['mean'] for m in keys]
    ax[1,0].bar(pos-.18,coverage,.36,label='全部类别');ax[1,0].bar(pos+.18,novel_cov,.36,label='复合新类')
    ax[1,0].set_xticks(pos,[NAMES[m] for m in keys],rotation=30,fontsize=7);ax[1,0].set(title='确认覆盖率：5-shot',ylabel='比例',ylim=(0,1));ax[1,0].legend(fontsize=8)
    u=analysis['unknown_summary'];us=[m for m in keys if f'multisignal_{m}' in u];up=np.arange(len(us))
    ax[1,1].bar(up-.18,[u[f'multisignal_{m}']['unknown_false_acceptance']['mean'] for m in us],.36,label='未知错误接受')
    ax[1,1].bar(up+.18,[u[f'multisignal_{m}']['known_coverage']['mean'] for m in us],.36,label='已知覆盖')
    ax[1,1].set_xticks(up,[NAMES[m] for m in us],rotation=30,fontsize=7);ax[1,1].set(title='K=0：未注册复合拒识',ylabel='比例',ylim=(0,1));ax[1,1].legend(fontsize=8)
    ax[1,2].axis('off');cells=[]
    for method in keys:
        s=summary[f'multisignal_{method}_k5']
        err=s['selective_error']['mean'];seq=s['sequential_error']['mean']
        cells.append([NAMES[method],'无确认' if err is None else f'{100*err:.1f}%',f"{100*s['sequential_coverage']['mean']:.1f}%",
            '无确认' if seq is None else f'{100*seq:.1f}%',f"{s['sequential_mean_windows']['mean']:.2f}"])
    table=ax[1,2].table(cellText=cells,colLabels=['方法','单窗确认错误','序贯覆盖','序贯错误','平均窗数'],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.7);ax[1,2].set_title('继续采样/拒判：平均最多四窗')
    fig3,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True);figures.append(fig3)
    fig3.canvas.manager.set_window_title('LASPI：测试信号频谱与训练')
    fig3.suptitle('实测信号频谱示例；幅值分别归一化，不能当作损伤严重度或电蚀证据')
    wave=np.load(Path(config['cache'])/'waveform_examples.npy',mmap_mode='r');freq=np.fft.rfftfreq(4096,1/12800)
    if chosen:
        qi=np.array(chosen['query_indices']);sample_labels=np.array([int(r['label']) for r in sample_rows])
        for j,c in enumerate((0,5,6)):
            idx=qi[np.flatnonzero(sample_labels[qi]==c)[0]];record=int(idx)//31
            for channel,label in ((3,'振动'),(0,'电流Ia原始输出'),(4,'电压Va原始输出')):
                signal=np.array(wave[record,channel]);amp=abs(np.fft.rfft((signal-signal.mean())*np.hanning(4096)));amp/=max(amp.max(),1e-12)
                axs[0,j].plot(freq,amp,label=label,linewidth=.8)
            axs[0,j].set(title=labels[c]+'：首测试记录',xlim=(0,1000),xlabel='频率（Hz）',ylabel='归一化FFT幅值');axs[0,j].legend(fontsize=7)
    first=config['scenarios'][0];seed=config['seeds'][0]
    for modality,color in (('vibration','C0'),('multisignal','C1')):
        path=reference/'models'/f'{first}_{modality}_s{seed}'/'encoder_history.json'
        if path.exists():
            h=json.loads(path.read_text(encoding='utf-8'));axs[1,0].plot([r['epoch'] for r in h],[r['source_val_nll'] for r in h],label='振动' if modality=='vibration' else '七通道',color=color)
    axs[1,0].set(title='编码器源验证NLL：固定首任务',xlabel='训练轮次',ylabel='NLL');axs[1,0].legend(fontsize=8)
    for head,label in (('deterministic','确定性'),('bayesian','贝叶斯')):
        path=reference/'models'/f'{first}_multisignal_s{seed}'/head/'head_history.json'
        if path.exists():
            h=json.loads(path.read_text(encoding='utf-8'));axs[1,1].plot([r['epoch'] for r in h],[r['source_val_nll'] for r in h],label=label)
    axs[1,1].set(title='特征头源验证NLL',xlabel='训练轮次',ylabel='NLL');axs[1,1].legend(fontsize=8)
    axs[1,2].axis('off');axs[1,2].text(0,.98,'频率与不确定性的边界\n\n分析采样率12.8kHz，FFT频率间隔3.125Hz。\nSTFT频率间隔12.5Hz；已滤除6.4kHz以上信息。\n\n电流/电压为电机三相传感器原始输出，\n没有当成轴承端口电学量。\n\nMI只涵盖Bayesian特征头的权重不确定性。\n低MI不证明未知工况安全，\n期望熵也不是已标定的电蚀噪声。',va='top',fontsize=10)
    if block:plt.show()
    return figures


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('result',type=Path);args=ap.parse_args();show_results(args.result)
