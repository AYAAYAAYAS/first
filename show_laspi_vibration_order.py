"""振动阶次实验中文交互图窗，仅手动保存。"""
import argparse, json, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'matplotlib_cache'))
import matplotlib.pyplot as plt
from prepare_laspi import CN


def show_base(result, config, summary, draw_only=False):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    records = json.loads((result/'metrics.json').read_text(encoding='utf-8'))
    names = {'deterministic':'确定性头','bayesian':'贝叶斯头'}
    fig, axes = plt.subplots(1,3,figsize=(15,5)); fig.canvas.manager.set_window_title('仅五基类：跨工况识别与校准')
    fig.suptitle(f'五基类：{config["axis"]}轴，编码器冻结={config["unfreeze"] == "none"}，不含复合原型竞争')
    for ax,key,title in zip(axes,('accuracy','macro_f1','ece'),('目标工况准确率','目标工况Macro-F1','目标校准ECE')):
        for method in names:
            selected = [r for r in records if r['method']==method]
            values = [np.mean([r['calibrated'][key] for r in selected if r['condition']==case]) for case in config['cases']]
            ax.plot(config['cases'],values,marker='o',label=names[method])
        ax.set(title=title,ylim=(0,1)); ax.tick_params(axis='x',rotation=30); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(rect=(0,0,1,.92))
    fig2,axs = plt.subplots(1,2,figsize=(12,5)); fig2.canvas.manager.set_window_title('五基类混淆矩阵：固定首工况与首种子')
    for ax,method in zip(axs,names):
        rec = next(r for r in records if r['condition']==config['cases'][0] and r['seed']==config['seeds'][0] and r['method']==method)
        cm = np.array(rec['calibrated']['confusion'],dtype=float); cm /= np.maximum(cm.sum(-1,keepdims=True),1)
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=1); ax.set(title=names[method],xlabel='预测',ylabel='真实')
        ax.set_xticks(range(5),CN[:5],rotation=40,ha='right'); ax.set_yticks(range(5),CN[:5])
        for r in range(5):
            for c in range(5): ax.text(c,r,f'{cm[r,c]:.0%}',ha='center',va='center',color='white' if cm[r,c]>.6 else 'black')
    fig2.tight_layout()
    figures = [fig,fig2]
    if draw_only:
        for figure in figures: figure.canvas.draw()
        plt.close('all'); return len(figures)
    print('打开五基类中文结果图窗，可通过工具栏手动保存。',flush=True)
    plt.show()


def show(result, draw_only=False):
    result = Path(result)
    config = json.loads((result/'config.json').read_text(encoding='utf-8'))
    summary = json.loads((result/'summary.json').read_text(encoding='utf-8'))
    if config.get('base_only'): return show_base(result,config,summary,draw_only)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    names = {'deterministic':'确定性头','bayesian':'贝叶斯头'}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.canvas.manager.set_window_title('振动跨工况：1／5／10／15-shot结果')
    fig.suptitle(f'仅振动跨工况GFSL：{config["axis"]}轴，收缩={config["shrinkage"]}，解冻={config["unfreeze"]}（未见工况子集）')
    for ax, key, title in zip(axes.flat, ('accuracy','macro_f1','novel_f1','ece','coverage','selective_error'),
                             ('七类准确率','七类宏平均F1','复合类F1','校准误差ECE（越小越好）','确认覆盖率','确认样本错误率（越小越好）')):
        for method in names:
            selected = sorted([s for s in summary if s['method'] == method], key=lambda s:s['k'])
            ax.plot([s['k'] for s in selected], [np.nan if s[key] is None else s[key] for s in selected], marker='o', label=names[method])
        ax.set(title=title, xlabel='每个复合类支持窗口数', ylabel='指标值', ylim=(0,1), xticks=config['shots'])
        ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(rect=(0,0,1,.94))
    # 首工况、首种子、首支持抽样，展示全部shot；不按最好成绩挑选。
    target = config['cases'][0]; seed = config['seeds'][0]
    bundle = result/'predictions'/f'window_zscore_vibration_bayesian_{target}_s{seed}.npz'
    fig2, axes2 = plt.subplots(1, len(config['shots']), figsize=(5*len(config['shots']),5), squeeze=False)
    fig2.canvas.manager.set_window_title('贝叶斯分类：各shot固定示例混淆矩阵')
    fig2.suptitle(f'{target}，种子{seed}，首支持抽样（非全任务平均）')
    with np.load(bundle) as data:
        y = data['true_labels']
        for ax,k in zip(axes2[0],config['shots']):
            j = int(np.flatnonzero(data['shots'] == k)[0]); p = data['probabilities'][j]
            cm = np.zeros((7,7)); np.add.at(cm,(y,p.argmax(-1)),1)
            cm /= np.maximum(cm.sum(-1,keepdims=True),1)
            ax.imshow(cm,cmap='Blues',vmin=0,vmax=1)
            ax.set(title=f'{k}-shot',xlabel='预测类别',ylabel='真实类别')
            ax.set_xticks(range(7),CN,rotation=55,ha='right',fontsize=8); ax.set_yticks(range(7),CN,fontsize=8)
            for r in range(7):
                for c in range(7): ax.text(c,r,f'{cm[r,c]:.0%}',ha='center',va='center',fontsize=7,color='white' if cm[r,c]>.6 else 'black')
    fig2.tight_layout(rect=(0,0,1,.9))
    folder = result/'models'/f'vibration_{target}_s{seed}'
    fig3, axes3 = plt.subplots(1,3,figsize=(16,5)); fig3.canvas.manager.set_window_title('预训练与部分解冻：训练曲线')
    for history_file,name in [(folder/'encoder_history.json','编码器预训练'),(folder/'deterministic'/'head_history.json','确定性联合训练'),(folder/'bayesian'/'head_history.json','贝叶斯联合训练')]:
        history = json.loads(history_file.read_text(encoding='utf-8'))
        for ax,key,title in zip(axes3,('train_loss','base_val_accuracy','base_val_nll'),('训练损失','源验证准确率','源验证NLL（选取检查点）')):
            ax.plot([h['epoch'] for h in history],[h[key] for h in history],label=name)
            ax.set(title=title,xlabel='训练轮数'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig3.tight_layout()
    figures = [fig,fig2,fig3]
    examples = result/'axis_examples.npz'
    if examples.exists():
        with np.load(examples) as data:
            fig4, axs = plt.subplots(1,2,figsize=(13,5)); fig4.canvas.manager.set_window_title('振动时频信息：频率轴与阶次轴')
            for i,rpm in enumerate(data['rpm']):
                axs[0].plot(data['hz_axis'],data['hz_spectrum'][i],label=f'{rpm:.0f} rpm')
                axs[1].plot(data['input_axis'],data['input_spectrum'][i],label=f'{rpm:.0f} rpm')
            axs[0].set(title='轴承内圈：固定首窗口，原Hz轴',xlabel='频率（Hz）',ylabel='时频图时间均值：log(1+功率)')
            axs[1].set(title='相同窗口：模型输入轴',xlabel='阶次（频率／轴转频率）' if config['axis']=='order' else '频率（Hz）',ylabel='时频图时间均值：log(1+功率)')
            for ax in axs: ax.legend(); ax.grid(alpha=.2)
            fig4.suptitle('不同转速下的内圈样本；阶次来自文件名rpm恒速近似，非包络谱'); fig4.tight_layout()
            figures.append(fig4)
    if draw_only:
        for figure in figures: figure.canvas.draw()
        plt.close('all'); return len(figures)
    print('打开中文图窗；点击工具栏保存按钮可手动保存，不自动导出图片。',flush=True)
    plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('result',type=Path)
    show(ap.parse_args().result)
