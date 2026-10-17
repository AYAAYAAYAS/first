"""Paired, descriptive analysis; no fitting/tuning with target labels."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.special import softmax
from prepare_laspi import ROOT,dump,csv_out
from run_laspi_gfsl import load_samples,masks,metrics,sequence_indices,fixed_distances,fit_temperature
from bayesian_layers import uncertainty

METHOD_NAMES={'euclidean':'确定性欧氏原型','mahalanobis':'确定性收缩马氏','bayesian_mahalanobis':'贝叶斯收缩马氏',
              'encoder_mahalanobis':'原始CNN表示＋马氏',
              'conditioned_mahalanobis':'工况修正＋确定性马氏','bayesian_conditioned_mahalanobis':'工况修正＋贝叶斯马氏'}


def agg(values):
    values=[float(v) for v in values if v is not None]
    return {'mean':float(np.mean(values)) if values else None,'std':float(np.std(values,ddof=1)) if len(values)>1 else (0. if values else None),'n_task_seed':len(values)}


def average_precision(labels,score):
    labels=np.asarray(labels,bool);score=np.asarray(score);order=np.argsort(-score,kind='stable')
    truth=labels[order];values=score[order];ends=np.r_[np.flatnonzero(values[:-1]!=values[1:]),len(values)-1]
    tp=np.cumsum(truth)[ends];recall=tp/max(int(truth.sum()),1);precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def augment_unknown_ap(reference,directory,unknown,rows,y):
    torch.set_num_threads(2)
    output=[]
    for r in unknown:
        scenario,seed,modality,method=r['scenario'],r['seed'],r['modality'],r['method']
        train,val,pool,target,role=masks(rows,scenario);vi=np.flatnonzero(val);ti=np.flatnonzero(target)
        path=reference/'models'/f'{scenario}_{modality}_s{seed}'
        if method=='encoder_mahalanobis':z=torch.from_numpy(np.load(path/'encoder_features.npy')[None])
        else:
            head='bayesian' if method.startswith('bayesian') else 'deterministic'
            z=torch.from_numpy(np.load(path/head/'posterior_features.npy'))
            if 'conditioned' in method:
                fit=json.loads((directory/f'fit_{scenario}_{modality}_{head}_s{seed}.json').read_text(encoding='utf-8'))
                coefficient=torch.from_numpy(np.load(directory/f'coefficient_{scenario}_{modality}_{head}_s{seed}.npy'))
                u=np.array([[row['speed_setting_hz'],row['load_percent']] for row in rows])
                normalized=torch.tensor((u-np.array(fit['context_mean']))/np.array(fit['context_std']),dtype=z.dtype)
                z=z-torch.einsum('nf,mfd->mnd',normalized,coefficient)
        distance,_=fixed_distances(z,y,train,method!='euclidean');v=-distance[:,vi].numpy();q=-distance[:,ti].numpy()
        temperature=fit_temperature(v,y[vi]);_,_,_,vmi=uncertainty(softmax(v/temperature,axis=-1))
        _,_,_,qmi=uncertainty(softmax(q/temperature,axis=-1))
        radius=max(float(np.quantile(distance[:,vi].mean(0).min(-1).values.numpy(),.95)),1e-6)
        limit=max(float(np.quantile(vmi,.95)),1e-6);score=distance[:,ti].mean(0).min(-1).values.numpy()/radius
        combined=np.maximum(score,qmi/limit) if method.startswith('bayesian') else score
        output.append({**r,'distance_auprc':average_precision(y[ti]>=5,score),'distance_mi_auprc':average_precision(y[ti]>=5,combined),
                       'unknown_prevalence':float((y[ti]>=5).mean())})
    dump(directory/'unknown_with_ap.json',output);return output


def analyze(reference,correction=None):
    reference=Path(reference).resolve();config=json.loads((reference/'config.json').read_text(encoding='utf-8'))
    rows,y,_=load_samples(Path(config['cache']));metrics_files=[reference]
    if correction:metrics_files.append(Path(correction).resolve())
    cohorts=[];record_results=[];displays={};unknown=[];decoder=[]
    for directory in metrics_files:
        directory_config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        results=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        lookup={(r['scenario'],r['seed'],r['modality'],r['method'],r['k'],r['episode']):r for r in results}
        if (directory/'display.json').exists():displays.update(json.loads((directory/'display.json').read_text(encoding='utf-8')))
        if (directory/'unknown_with_ap.json').exists():extra_unknown=json.loads((directory/'unknown_with_ap.json').read_text(encoding='utf-8'))
        else:extra_unknown=augment_unknown_ap(reference,directory,json.loads((directory/'unknown.json').read_text(encoding='utf-8')),rows,y)
        unknown+=extra_unknown
        for path in sorted((directory/'predictions').glob('*.npz')):
            # Robust parsing from stored result keys, not filename splitting.
            name=path.stem;possible=[k for k in lookup if name==f'{k[0]}_{k[2]}_{k[3]}_s{k[1]}']
            if not possible:raise ValueError('Prediction bundle has no matching metric keys')
            scenario,seed,modality,method,_,_=possible[0]
            with np.load(path) as data:
                query=data['query_indices'];p=data['probabilities'];labels=data['true_labels'];selection=data['acceptance']
                group_map=defaultdict(list)
                for local,idx in enumerate(query):group_map[rows[int(idx)]['record_id']].append(local)
                groups=[np.array(group_map[name]) for name in sorted(group_map)]
                record_labels=np.array([labels[g[0]] for g in groups]);seq_groups=sequence_indices(rows,query)
                seq_ids=np.stack([a for _,a in seq_groups]);seq_truth=labels[seq_ids[:,0]]
                for j,(k,episode) in enumerate(zip(data['shots'],data['episodes'])):
                    r=lookup[(scenario,seed,modality,method,int(k),int(episode))]
                    display_key=f'{modality}_{method}'
                    if display_key not in displays and scenario==config['scenarios'][0] and seed==config['seeds'][0] and int(k)==5 and int(episode)==0:
                        displays[display_key]={**r,'query_indices':query.tolist(),'probabilities':p[j].tolist(),
                            'mi':data['uncertainties'][j,:,2].tolist(),'entropy':data['uncertainties'][j,:,0].tolist(),
                            'accepted':selection[j,:,1].tolist()}
                    record_p=np.stack([p[j,g].mean(0) for g in groups]);record_metrics=metrics(record_labels,record_p)
                    fixed_four=p[j,seq_ids].mean(1).argmax(1)
                    row={'scenario':scenario,'seed':seed,'modality':modality,'method':method,'k':int(k),'episode':int(episode),
                         'record_macro_f1':record_metrics['macro_f1'],'record_novel_f1':record_metrics['novel_f1'],
                         'fixed_four_error':float((fixed_four!=seq_truth).mean()),
                         'known_coverage':float(selection[j,labels<5,1].mean()),'novel_coverage':float(selection[j,labels>=5,1].mean())}
                    record_results.append(row)
                    accepted=selection[j,:,1]
                    which=labels>=5;correct=p[j].argmax(1)==labels
                    for label_name,mask in (('known',~which),('novel',which)):
                        keep=accepted & mask
                        row[label_name+'_selective_error']=float(1-correct[keep].mean()) if keep.any() else None
                    record_dict={key:float(r['calibrated'][key]) for key in ('accuracy','macro_f1','base_f1','novel_f1','harmonic_f1','nll','ece','brier','health_accuracy')}
                    record_dict.update(raw_ece=r['raw']['ece'],raw_nll=r['raw']['nll'],
                        coverage=r['selective_full']['coverage'],selective_error=r['selective_full']['selective_error'],
                        correct_confirmed_fraction=r['selective_full']['correct_confirmed_fraction'],
                        sequential_coverage=r['sequential']['coverage'],sequential_error=r['sequential']['selective_error'],
                        sequential_mean_windows=r['sequential']['mean_windows'],**{key:value for key,value in row.items() if key not in
                            ('scenario','seed','modality','method','k','episode')})
                    cohorts.append({'scenario':scenario,'seed':seed,'modality':modality,'method':method,'k':int(k),'episode':int(episode),**record_dict})
        # release potentially large trajectory lists before next result directory
        del results,lookup
    csv_out(reference/'record_and_rejection_metrics.csv',record_results)
    group=defaultdict(list)
    for r in cohorts:group[(r['scenario'],r['seed'],r['modality'],r['method'],r['k'])].append(r)
    task_seed=[];fields=[k for k in cohorts[0] if k not in ('scenario','seed','modality','method','k','episode')]
    for key,items in sorted(group.items()):
        row=dict(zip(('scenario','seed','modality','method','k'),key))
        for field in fields:
            values=[r[field] for r in items if r[field] is not None]
            row[field]=float(np.mean(values)) if values else None
        task_seed.append(row)
    csv_out(reference/'task_seed_summary.csv',task_seed)
    summary={};by_scenario={}
    for modality in sorted({r['modality'] for r in task_seed}):
        for method in sorted({r['method'] for r in task_seed}):
            for k in config['shots']:
                subset=[r for r in task_seed if r['modality']==modality and r['method']==method and r['k']==k]
                if not subset:continue
                summary[f'{modality}_{method}_k{k}']={field:agg([r[field] for r in subset]) for field in fields}
                for scenario in config['scenarios']:
                    selected=[r for r in subset if r['scenario']==scenario]
                    by_scenario[f'{scenario}_{modality}_{method}_k{k}']={field:agg([r[field] for r in selected]) for field in fields}
    paired={};index={(r['scenario'],r['seed'],r['modality'],r['method'],r['k']):r for r in task_seed}
    for modality in sorted({r['modality'] for r in task_seed}):
        for k in config['shots']:
            for left,right in [('bayesian_mahalanobis','mahalanobis'),('conditioned_mahalanobis','mahalanobis'),
                               ('bayesian_conditioned_mahalanobis','bayesian_mahalanobis'),('bayesian_conditioned_mahalanobis','conditioned_mahalanobis')]:
                available=[]
                for scenario in config['scenarios']:
                    for seed in config['seeds']:
                        a=index.get((scenario,seed,modality,left,k));b=index.get((scenario,seed,modality,right,k))
                        if a and b:available.append((a,b))
                if available:paired[f'{modality}_k{k}_{left}_minus_{right}']={field:agg([a[field]-b[field] for a,b in available
                    if a[field] is not None and b[field] is not None]) for field in ('macro_f1','novel_f1','nll','ece','coverage')}
    unknown_summary={}
    for modality in sorted({r['modality'] for r in unknown}):
        for method in sorted({r['method'] for r in unknown}):
            subset=[r for r in unknown if r['modality']==modality and r['method']==method]
            if subset:unknown_summary[f'{modality}_{method}']={field:agg([r[field] for r in subset]) for field in
                ('distance_auroc','distance_mi_auroc','distance_auprc','distance_mi_auprc','unknown_prevalence','unknown_false_acceptance','known_coverage')}
    # Separate encoder transfer failure from adding two novel competitors.
    models=json.loads((reference/'models.json').read_text(encoding='utf-8'))
    for item in models:
        if item['head']!='deterministic':continue
        scenario,seed,modality=item['scenario'],item['seed'],item['modality']
        train,val,pool,target,_=masks(rows,scenario);q=np.flatnonzero(target & (y<5))
        path=reference/'models'/f'{scenario}_{modality}_s{seed}'
        features=np.load(path/'encoder_features.npy');state=torch.load(path/'best_encoder.pt',weights_only=True,map_location='cpu')
        logits=features[q]@state['classifier.weight'].numpy().T+state['classifier.bias'].numpy()
        history=json.loads((path/'encoder_history.json').read_text(encoding='utf-8'))
        chosen=min(history,key=lambda r:r['source_val_nll'])
        decoder.append({'scenario':scenario,'seed':seed,'modality':modality,
            'source_known_validation_accuracy':chosen['source_val_accuracy'],
            'target_known_only_accuracy':float((logits.argmax(1)==y[q]).mean()),
            'interpretation':'Five known candidates only; target transfer diagnostic, not seven-class GFSL score'})
    analysis={'summary':summary,'by_scenario':by_scenario,'paired_deltas':paired,'unknown_summary':unknown_summary,
        'encoder_transfer':decoder,'correction_directory':str(Path(correction).resolve()) if correction else None,
        'summary_unit':'20 support draws averaged per task/seed, then descriptive task/seed mean and SD; no specimen-population CI',
        'query_windows_reused_across_task_families':True,'target_labels_used_for_posthoc_evaluation_only':True,
        'condition_correction_is_exploratory':bool(correction)}
    dump(reference/'analysis.json',analysis);dump(reference/'dashboard_display.json',displays)
    write_report(reference,config,analysis);print('ANALYSIS_COMPLETE',reference,flush=True);return analysis


def write_report(reference,config,analysis):
    lines=['# '+config['title'],'','第一篇采用LASPI无载流机理约束的诊断实验；机理建模及物理约束留到第二篇。',
      '','## 数据审核','',
      '原始336个CSV中发现5组完全相同的SHA256：4组分别位于半齿断裂与半齿断裂＋外圈目录（25Hz/75%），1组健康信号同时位于35Hz/50%和35Hz/75%。正式实验删除这些记录的使用资格，全部10个副本均排除；原始文件未删除、未自行修正标签。有效326个CSV、10106个窗口，部分工况类别覆盖不再完整。',
      '','## 协议与预算','',
      '- 整状态—工况文件夹内的采集文件不跨集合；未确认实体轴承或四个文件是否连续采集，不能称跨独立轴承。',
      '- 三个留转速任务、四个留载荷任务；五个健康/单故障基类用于源训练和源验证，两类复合仅从源训练工况各抽K个支持。K=1/5/10，各20组；源支持与目标查询隔离，不是目标域少样本适应。',
      '- 振动与七通道两种输入，三种子。每个输入复用同一已训练编码器比较确定性/贝叶斯头；另有欧氏与马氏距离比较。Bayesian平均场Gaussian权重后验，先验N(0,1)，训练MC2、评价MC20。',
      '- 原25.6kHz数据抗混叠重采样至12.8kHz，每窗4096点/0.32s；每记录31个无重叠窗，剩余0.08s不用。STFT n_fft1024、hop256、center=False；滤除6.4kHz以上信息是共同预处理限制。原始传感器增益不用于物理计算。',
      '- 25轮编码器训练、每轮20个平衡批次；25轮特征头、每轮10个episode。轮次是固定批次数，不是全部窗口的一次遍历。标签平滑0.05，KL/P系数0.01属于带权变分判别代理目标，不是全管线精确Bayesian后验。',
      '- 归一化、模型选择、温度及拒判阈值只用源基类。温度范围0.1–20；目标错误率5%是源验证经验标准，不能解释为目标工况风险保证。',
      '- 工况修正为看到早期域迁移失败后增加的探索性对照：只对源基类工况组均值拟合共享线性转速/载荷效应，固定ridge1，使用测试运行设定作输入，不使用目标故障标签或重新训练网络。此结果仍需独立确认。',
      '','## 跨工况GFSL结果','',
      '先平均每个任务/种子的支持抽样，再汇总21个任务/种子；SD为技术/任务波动，不是独立轴承置信区间。以下5-shot均为七类目标查询。',
      '','| 输入 | 方法 | Macro-F1 | 复合F1 | ECE（原始→校准） | NLL（原始→校准） | 确认覆盖 | 确认错误 | 复合覆盖 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def value(s,key,pct=False):
        v=s[key]['mean'];return '无确认' if v is None else (f'{100*v:.2f}%' if pct else f'{v:.4f}')
    for key,s in analysis['summary'].items():
        if not key.endswith('_k5'):continue
        modality='七通道' if key.startswith('multisignal_') else '振动'
        method=key[len('multisignal_'):] if modality=='七通道' else key[len('vibration_'):]
        method=method[:-3]
        lines.append('| '+modality+' | '+METHOD_NAMES[method]+' | '+value(s,'macro_f1',True)+' | '+value(s,'novel_f1',True)+' | '+
          value(s,'raw_ece')+'→'+value(s,'ece')+' | '+value(s,'raw_nll')+'→'+value(s,'nll')+' | '+value(s,'coverage',True)+' | '+
          value(s,'selective_error',True)+' | '+value(s,'novel_coverage',True)+' |')
    lines+=['','## 未注册复合故障拒识（K=0）','','| 输入 | 方法 | 距离AUROC | 距离＋MI AUROC | 未知错误接受率 | 已知覆盖 |',
            '|---|---|---:|---:|---:|---:|']
    for key,s in analysis['unknown_summary'].items():
        modality='七通道' if key.startswith('multisignal_') else '振动'
        method=key[len('multisignal_'):] if modality=='七通道' else key[len('vibration_'):]
        lines.append('| '+modality+' | '+METHOD_NAMES[method]+' | '+value(s,'distance_auroc')+' | '+value(s,'distance_mi_auroc')+' | '+
                     value(s,'unknown_false_acceptance',True)+' | '+value(s,'known_coverage',True)+' |')
    lines+=['','## 解释边界','','注册后的拒判是选择性分类；K=0是未注册复合检测；留转速/载荷是工况迁移。三者不是同一个指标。',
        '降低确认错误而几乎全部拒判不能称诊断有效，尤其应查看两类复合的覆盖率。温度校准不保证目标工况ECE或NLL改善。贝叶斯权重MI只表达特征头的参数不确定性，不包括冻结编码器、支持集全部统计误差或未知故障机理。',
        '窗口级指标和文件平均后验分类分别保存，文件平均也不等于独立轴承验证。固定四窗与序贯确认使用相同连续窗口；窗口后验取平均，不假设相邻窗独立并相乘。',
        '如结果为负，不据此更改测试权重、选择最佳任务或种子，亦不能推断所有GFSL/BNN方法无效。当前结果属于固定公开台架上的方法开发和探索验证；投稿前仍需新模型对照和独立数据。',
        '','## 文件与运行','',
        '`config.json`、`source_snapshot/`保存配置与执行代码；`models/`保存最佳/最后编码器和特征头、每个后验抽样权重及可重放特征；`support_ids.json`保存真实支持预算；`predictions/*.npz`保存每组查询概率、熵/MI及接受掩码；`task_seed_summary.csv`和`analysis.json`提供分任务及配对变化。',
        '图窗中文，训练结束可打开并手动保存；程序不自动保存图片。当前结果不涉及电蚀机理、连续健康指数或寿命。',
        '','## 数据来源','',
        'Soualhi等，Open Heterogeneous Data for Condition Monitoring of Multi Faults in Rotating Machines Used in Different Operating Conditions. IJPHM, 2023, DOI:10.36001/ijphm.2023.v14i2.3497，原文第7–8及14页。数据来源为用户提供的LASPI官方公开包。',
        '公开数据的故障是预制缺陷，并非已验证的电蚀退化标签；本篇题目和结论相应限定为齿轮—轴承复合故障诊断。','']
    (reference/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('reference',type=Path);ap.add_argument('--correction',type=Path)
    args=ap.parse_args();analyze(args.reference,args.correction)
