"""Five known candidates only: diagnose representation before novel competition."""
import json,sys,collections
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from prepare_laspi import ROOT,dump
from run_laspi_gfsl import load_samples,fit_temperature
from run_laspi_same_condition import split_files
from mahalanobis_metric import fit_reference,whiten


def fixed_coverage_risk(result,config,conditions):
    """Descriptive test ranking at the same coverage; not a deployable risk guarantee."""
    output=[]
    for norm in config['normalizations']:
        for modality in config['modalities']:
            for method in ('deterministic','bayesian'):
                risks={.6:[],.8:[],1.:[]}
                for seed in config['seeds']:
                    for condition in conditions:
                        file=result/'predictions'/f'{norm}_{modality}_{method}_{condition}_s{seed}.npz'
                        if not file.exists():continue
                        with np.load(file) as n:
                            p=n['probabilities'][n['shots']==5];truth=n['true_labels']
                        if not len(p):continue
                        correct=p.argmax(-1)==truth
                        ordered=np.argsort(-p.max(-1),axis=-1,kind='stable')
                        ranked=np.take_along_axis(correct,ordered,axis=-1)
                        for level in risks:risks[level].append(float((1-ranked[:,:int(np.ceil(level*len(truth)))].mean(-1)).mean()))
                if risks[.6]:
                    output.append({'normalization':norm,'modality':modality,'method':method,'k':5,
                        'ranking':'maximum class probability only; descriptive test comparison',
                        **{f'risk_at_{int(level*100)}_percent_coverage':float(np.mean(v)) for level,v in risks.items()},
                        'condition_seed_count':len(risks[.6])})
    dump(result/'risk_at_fixed_coverage.json',output)
    return output


def known_only(result):
    torch.set_num_threads(2);result=Path(result);config=json.loads((result/'config.json').read_text(encoding='utf-8'))
    rows,y,_=load_samples(ROOT/'prepared');records=[]
    for norm in config['normalizations']:
        for modality in config['modalities']:
            for seed in config['seeds']:
                role,cond,conditions,_=split_files(rows,seed)
                if config['conditions']:conditions=[c for c in conditions if c in config['conditions']]
                for head in ('deterministic','bayesian'):
                    folder=result/'models'/f'{norm}_{modality}_s{seed}'/head
                    if not (folder/'posterior_features.npy').exists():continue
                    z=np.load(folder/'posterior_features.npy')
                    for name in conditions:
                        tr=np.flatnonzero((role=='base_train')&(cond==name));va=np.flatnonzero((role=='base_val')&(cond==name));qu=np.flatnonzero((role=='query')&(cond==name)&(y<5))
                        validation=[];query=[]
                        with torch.no_grad():
                            for sample in z:
                                zz=torch.as_tensor(sample);means,chol=fit_reference(zz[tr],torch.as_tensor(y[tr]));w=whiten(zz,chol).numpy().astype(np.float64);m=whiten(means,chol).numpy().astype(np.float64)
                                def score(ids):return -np.maximum((w[ids]**2).sum(-1)[:,None]+(m**2).sum(-1)[None]-2*w[ids]@m.T,0)/z.shape[-1]
                                validation.append(score(va));query.append(score(qu))
                        temperature=fit_temperature(np.stack(validation),y[va]);p=softmax(np.stack(query)/temperature,axis=-1).mean(0);pred=p.argmax(-1)
                        cm=np.bincount(5*y[qu]+pred,minlength=25).reshape(5,5);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
                        records.append({'normalization':norm,'modality':modality,'method':head,'seed':seed,'condition':name,
                            'accuracy':float((pred==y[qu]).mean()),'macro_f1':float(f1.mean()),'confusion':cm.tolist(),'temperature':temperature,
                            'nll':float(-np.log(p[np.arange(len(qu)),y[qu]].clip(1e-12)).mean()),'scope':'five known candidates only; query files held out'})
    grouped=collections.defaultdict(list)
    for r in records:grouped[(r['normalization'],r['modality'],r['method'])].append(r)
    summary=[{**dict(zip(('normalization','modality','method'),key)),**{m:float(np.mean([r[m] for r in v])) for m in ('accuracy','macro_f1','nll')},'condition_seed_count':len(v)} for key,v in sorted(grouped.items())]
    dump(result/'known_only_metrics.json',records);dump(result/'known_only_summary.json',summary)
    risk=fixed_coverage_risk(result,config,conditions)
    report=result/'EXPERIMENT_REPORT.md'
    if report.exists():
        body=report.read_text(encoding='utf-8').split('\n## 五个已知候选的测试诊断')[0]
        lines=['','## 五个已知候选的测试诊断','','仅测试五基类文件，暂不让复合类参与竞争；用于定位特征学习和七类边界问题，不是完整GFSL得分。',
               '','|处理|输入|头|准确率|Macro-F1|NLL|','|---|---|---|---:|---:|---:|']
        for s in summary:lines.append(f"|{s['normalization']}|{s['modality']}|{s['method']}|{s['accuracy']:.2%}|{s['macro_f1']:.2%}|{s['nll']:.4f}|")
        lines.extend(['','## 相同覆盖率的风险比较','','5-shot测试中，按最大类别概率排序，各工况/种子接受最高约60%或80%，支持抽样先平均。仅为测试描述性比较，未用测试标签设定部署阈值，不是风险上界。',
            '','|处理|输入|头|60%覆盖时错误率|80%覆盖时错误率|','|---|---|---|---:|---:|'])
        for s in risk:lines.append(f"|{s['normalization']}|{s['modality']}|{s['method']}|{s['risk_at_60_percent_coverage']:.2%}|{s['risk_at_80_percent_coverage']:.2%}|")
        report.write_text(body+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    print('KNOWN_ONLY',json.dumps(summary,ensure_ascii=False));return summary


if __name__=='__main__':known_only(sys.argv[1])
