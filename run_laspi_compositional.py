"""成分感知Compositional Prototype；复用模型，同支持集三方案配对比较。

alpha：以K个给定支持的均值，正则化投影到两个基类原型连线。
lambda：K/(K+prior_strength)，也可用--support-weight指定固定lambda。
没有真实复合查询参与alpha/lambda拟合；此版没有额外门控网络。
"""
import argparse, json, sys, shutil, collections, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.special import softmax
from prepare_laspi import ROOT, CN, dump, csv_out, sha256
from run_laspi_gfsl import (load_samples, fit_temperature, metrics, threshold_confidence,
                            selective_metrics, save_predictions)
from run_laspi_cross_702010 import split_cross, cross_supports
from mahalanobis_metric import fit_reference, whiten
from bayesian_layers import uncertainty

# LASPI真实标签映射：不是Inner+Outer，也不存在Ball基类。
COMPONENTS = {5:(1,3), 6:(2,4)}
SCHEMES = ('support_only','fixed_composition','adaptive_composition')


def compose(base, support, pair, scheme, prior_strength=2., alpha_regularization=1., support_weight=None):
    """输入已按源协方差白化；每个BNN后验抽样分别计算原型和alpha。"""
    center = support.mean(1); a,b = pair
    alpha = np.full(len(base), .5)
    if scheme == 'adaptive_composition':
        direction = base[:,a]-base[:,b]
        # argmin ||center-[alpha*Pa+(1-alpha)*Pb]||^2 + gamma*(alpha-.5)^2
        numerator = (direction*(center-base[:,b])).sum(-1)+alpha_regularization*.5
        denominator = (direction*direction).sum(-1)+alpha_regularization
        alpha = np.clip(numerator/np.maximum(denominator,1e-12),0,1)
    prior = alpha[:,None]*base[:,a]+(1-alpha[:,None])*base[:,b]
    k = support.shape[1]
    weight = 1. if scheme == 'support_only' else (k/(k+prior_strength) if support_weight is None else support_weight)
    prototype = (1-weight)*prior+weight*center
    return prototype, alpha, float(weight)


def distance_logits(values, means):
    squared = (values*values).sum(-1)[:,:,None]+(means*means).sum(-1)[:,None]-2*np.einsum('mnd,mcd->mnc',values,means)
    return -np.maximum(squared,0)/values.shape[-1]


def evaluate(z,rows,y,role,target,seed,method,args,source_config,result):
    tr=np.flatnonzero(role=='base_train');va=np.flatnonzero(role=='base_val');query=np.flatnonzero(role=='query')
    whitened=[];references=[]
    with torch.no_grad():
        for draw in z:
            x=torch.as_tensor(draw);means,chol=fit_reference(x[tr],torch.as_tensor(y[tr]),shrinkage=source_config['shrinkage'])
            whitened.append(whiten(x,chol).numpy());references.append(whiten(means,chol).numpy())
    w=np.stack(whitened).astype(np.float64);base=np.stack(references).astype(np.float64)
    blocks={s:[] for s in SCHEMES};uncertainties={s:[] for s in SCHEMES};acceptance={s:[] for s in SCHEMES}
    alpha_blocks={s:[] for s in SCHEMES};weights={s:[] for s in SCHEMES};shot_ids=[];records=[];support_manifest=[]
    for k in args.shots:
        for support in cross_supports(rows,y,role,target,seed,k,args.episodes):
            support_manifest.append(support);shot_ids.append(k)
            for scheme in SCHEMES:
                novel=[];alphas=[];lambdas=[]
                for label,ids in zip((5,6),support['indices']):
                    p,alpha,weight=compose(base,w[:,ids],COMPONENTS[label],scheme,args.prior_strength,args.alpha_regularization,args.support_weight)
                    novel.append(p[:,None]);alphas.append(alpha);lambdas.append(weight)
                means=np.concatenate([base]+novel,axis=1)
                vl=distance_logits(w[:,va],means);ql=distance_logits(w[:,query],means)
                temperature=fit_temperature(vl,y[va])
                v,_,_,vmi=uncertainty(softmax(vl/temperature,axis=-1))
                p,h,eh,mi=uncertainty(softmax(ql/temperature,axis=-1));raw=softmax(ql,axis=-1).mean(0)
                threshold=threshold_confidence(y[va],v,source_config.get('target_error',.05))
                radius=float(np.quantile(-vl.mean(0)[np.arange(len(va)),y[va]],.95));mi_limit=float(np.quantile(vmi,.95))
                pred=p.argmax(-1);accepted=(p.max(-1)>=threshold)&(-ql.mean(0)[np.arange(len(query)),pred]<=radius)
                if method=='bayesian':accepted &= mi<=mi_limit
                record={'condition':target,'seed':seed,'method':method,'scheme':scheme,'k':k,'episode':support['episode'],
                        'calibrated':metrics(y[query],p),'raw':metrics(y[query],raw),
                        'selective':selective_metrics(pred==y[query],accepted),'temperature':temperature,
                        'component_alpha_mean':np.array(alphas).mean(1).tolist(),'support_weight':lambdas,
                        'base_to_novel_error':float((pred[y[query]<5]>=5).mean()),
                        'confidence_threshold':threshold,'radius':radius,'mi_limit':mi_limit}
                records.append(record);blocks[scheme].append(p);uncertainties[scheme].append(np.column_stack([h,eh,mi]))
                acceptance[scheme].append(accepted);alpha_blocks[scheme].append(np.stack(alphas,axis=-1));weights[scheme].append(lambdas)
        for scheme in SCHEMES:
            selected=[r for r in records if r['k']==k and r['scheme']==scheme]
            print(f'{target} s{seed} {method} {k}-shot {scheme}：MacroF1={np.mean([r["calibrated"]["macro_f1"] for r in selected]):.4f} 基类F1={np.mean([r["calibrated"]["base_f1"] for r in selected]):.4f} 复合F1={np.mean([r["calibrated"]["novel_f1"] for r in selected]):.4f} lambda={selected[0]["support_weight"][0]:.3f} alpha={np.round(np.mean([r["component_alpha_mean"] for r in selected],axis=0),3).tolist()}',flush=True)
    prefix=f'{method}_{target}_s{seed}'
    dump(result/'predictions'/f'{prefix}_supports.json',support_manifest)
    for scheme in SCHEMES:
        save_predictions(result/'predictions'/f'{prefix}_{scheme}.npz',probabilities=np.stack(blocks[scheme]),
                         uncertainties=np.stack(uncertainties[scheme]),acceptance=np.stack(acceptance[scheme]),
                         shots=np.array(shot_ids),true_labels=y[query],query_indices=query,
                         component_alpha=np.stack(alpha_blocks[scheme]),support_weight=np.array(weights[scheme]))
    return records


def summarize(result,records):
    groups=collections.defaultdict(list)
    for r in records:groups[(r['method'],r['scheme'],r['k'],r['condition'],r['seed'])].append(r)
    task=[]
    for key,items in sorted(groups.items()):
        r=dict(zip(('method','scheme','k','condition','seed'),key))
        for name in ('accuracy','macro_f1','base_f1','novel_f1','ece','nll'):r[name]=float(np.mean([x['calibrated'][name] for x in items]))
        r['raw_ece']=float(np.mean([x['raw']['ece'] for x in items]));r['base_to_novel_error']=float(np.mean([x['base_to_novel_error'] for x in items]))
        task.append(r)
    csv_out(result/'task_seed_summary.csv',task);summary=[]
    for key in sorted(set((r['method'],r['scheme'],r['k']) for r in task)):
        selected=[r for r in task if (r['method'],r['scheme'],r['k'])==key]
        summary.append({**dict(zip(('method','scheme','k'),key)),**{name:float(np.mean([x[name] for x in selected])) for name in ('accuracy','macro_f1','base_f1','novel_f1','ece','raw_ece','nll','base_to_novel_error')},'task_seed_count':len(selected)})
    dump(result/'summary.json',summary)
    lines=['# 成分感知原型：配对对照','','复用同一训练模型、CSV分区、同一复合支持集。五基类原型和共享马氏协方差不变；两复合类仅使用指定K支持标签。',
           '组合映射：齿面+内圈(1,3)，半齿断裂+外圈(2,4)。组合先验是统计假设，不代表已证明的物理叠加。',
           'support_only：原支持均值；fixed_composition：alpha=0.5；adaptive_composition：支持均值正则化投影确定alpha，没有额外门控网络。',
           '默认lambda=K/(K+prior_strength)；可固定lambda。该权重可在高斯工作假设下理解为均值收缩，不宣称已验证真实概率先验；支持CSV不等于独立试件。',
           '查询标签仅用于评价；源验证仅校准，不自动选择alpha/lambda超参数。BNN权重不再更新，本实验为原型构建消融。','',
           '|头|原型方案|K|Macro-F1|基类F1|复合F1|原始ECE|校准ECE|基类误判复合|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['method']}|{r['scheme']}|{r['k']}|{r['macro_f1']:.2%}|{r['base_f1']:.2%}|{r['novel_f1']:.2%}|{r['raw_ece']:.4f}|{r['ece']:.4f}|{r['base_to_novel_error']:.2%}|")
    (result/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def show(result):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib_cache'))
    import matplotlib.pyplot as plt
    summary=json.loads((result/'summary.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    names={'support_only':'普通支持原型','fixed_composition':'固定成分组合','adaptive_composition':'自适应成分组合'}
    methods=[method for method in ('deterministic','bayesian') if any(r['method']==method for r in summary)]
    fig,axs=plt.subplots(len(methods),3,figsize=(16,4.5*len(methods)),squeeze=False);fig.canvas.manager.set_window_title('成分感知原型：同支持集配对对照')
    for row,method in enumerate(methods):
        for ax,key,title in zip(axs[row],('macro_f1','novel_f1','base_f1'),('七类Macro-F1','复合类F1','五基类F1')):
            for scheme in SCHEMES:
                selected=sorted([r for r in summary if r['method']==method and r['scheme']==scheme],key=lambda r:r['k'])
                if selected:ax.plot([r['k'] for r in selected],[r[key] for r in selected],marker='o',label=names[scheme])
            ax.set(title=method+'：'+title,xlabel='每个复合类支持窗口数',ylim=(0,1));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();plt.show()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-result',type=Path,default=ROOT/'results'/'vibration_order_mixed15')
    ap.add_argument('--output',type=Path);ap.add_argument('--cases',nargs='+');ap.add_argument('--seeds',nargs='+',type=int)
    ap.add_argument('--shots',nargs='+',type=int,default=[1,5,10,15,20]);ap.add_argument('--episodes',type=int,default=15)
    ap.add_argument('--methods',nargs='+',choices=('deterministic','bayesian'),default=['deterministic','bayesian'])
    ap.add_argument('--prior-strength',type=float,default=2.);ap.add_argument('--alpha-regularization',type=float,default=1.)
    ap.add_argument('--support-weight',type=float,help='固定lambda，省略时K/(K+prior-strength)')
    ap.add_argument('--no-show',action='store_true');args=ap.parse_args(argv)
    if min(args.shots)<1 or args.episodes<1 or args.prior_strength<0 or args.alpha_regularization<0:ap.error('shot/episodes须为正，正则系数非负')
    if args.support_weight is not None and not 0<=args.support_weight<=1:ap.error('support-weight须在[0,1]')
    source=args.model_result.resolve();config=json.loads((source/'config.json').read_text(encoding='utf-8'))
    if config.get('protocol')!='cross_condition_702010' or config.get('modalities')!=['vibration']:ap.error('需要振动跨工况入口的模型结果')
    cases=args.cases or config['cases'];seeds=args.seeds or config['seeds']
    if not set(cases)<=set(config['cases']) or not set(seeds)<=set(config['seeds']):ap.error('工况/种子必须已经完成训练')
    rows,y,_=load_samples(ROOT/'prepared_window_zscore',metadata_only=True)
    result=(args.output or ROOT/'results'/'compositional_prototype').resolve();original=result;number=2
    while result.exists():result=original.with_name(original.name+f'_run{number}');number+=1
    for name in ('predictions','splits','source_snapshot'):(result/name).mkdir(parents=True,exist_ok=True)
    sources=('run_laspi_compositional.py','run_laspi_cross_702010.py','mahalanobis_metric.py','run_laspi_gfsl.py','bayesian_layers.py')
    dump(result/'config.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'output':str(result),'source_config':config,
         'components':COMPONENTS,'cases':cases,'seeds':seeds,'schemes':SCHEMES,'code_hashes':{f:sha256(ROOT/f) for f in sources}})
    for f in sources:shutil.copy2(ROOT/f,result/'source_snapshot'/f)
    torch.set_num_threads(2);started=time.perf_counter();records=[]
    print('复用已训练编码器和头，不启动GPU训练；所有原型方案使用同一个支持集。结果：'+str(result),flush=True)
    for target in cases:
        for seed in seeds:
            role,partition,cond,conditions,audit=split_cross(rows,target,seed)
            import csv
            with (source/'splits'/f'{target}_s{seed}.csv').open(encoding='utf-8-sig') as f:original_split=list(csv.DictReader(f))
            if [(r['sample_id'],r['role']) for r in original_split]!=[(r['sample_id'],role[i]) for i,r in enumerate(rows)]:raise ValueError('与模型训练分区不一致')
            dump(result/'splits'/f'{target}_s{seed}.json',audit)
            for method in args.methods:
                path=source/'models'/f'vibration_{target}_s{seed}'/method/'posterior_features.npy'
                if not path.exists():raise FileNotFoundError('需先完成此模型分支：'+str(path))
                with path.open('rb') as stream:z=np.load(stream,allow_pickle=False)
                records.extend(evaluate(z,rows,y,role,target,seed,method,args,config,result))
                dump(result/'metrics.json',records);summarize(result,records)
    dump(result/'execution.json',{'finished':True,'elapsed_seconds':time.perf_counter()-started,'evaluations':len(records),'network_weights_updated':False})
    print('全部完成：'+str(result),flush=True)
    if not args.no_show:show(result)
    return result


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    main()
