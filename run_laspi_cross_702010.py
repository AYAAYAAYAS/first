"""CSV-count 70/20/10 LASPI split, with separately reported unseen-condition test.

Unseen speed/load combinations, all individual speed/load levels in training.
Novel source supports have exactly K labeled windows from K distinct CSV files.
"""
from __future__ import annotations
import argparse,collections,csv,itertools,json,math,os,shutil,time,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from scipy.special import softmax
from prepare_laspi import ROOT,dump,csv_out,sha256
from run_laspi_gfsl import (load_samples,image_scaler,fit_temperature,metrics,
    threshold_confidence,selective_metrics,save_predictions,MODALITIES)
from run_laspi_same_condition import split_files,train_encoder,train_head,summarize
from bayesian_layers import uncertainty
from mahalanobis_metric import fit_reference,whiten,mahalanobis_logits

TITLE='跨工况GFSL：按CSV数量70/20/10划分、归一化与统一马氏距离'
CASES=('25hz_0pct','35hz_50pct','45hz_75pct')


def split_cross(rows,target,seed):
    _,cond,complete,skipped=split_files(rows,42)
    if target not in complete:raise ValueError('Target condition is incomplete: '+target)
    y=np.array([r['label'] for r in rows]);active=~np.array([r['excluded_duplicate'] for r in rows])
    role=np.full(len(rows),'unassigned',dtype=object);partition=role.copy()
    role[~active]='excluded_duplicate';partition[~active]='excluded_duplicate'
    files=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if active[i]:files[r['record_id']].append(i)
    total=len(files);train_count=int(round(.7*total));val_count=int(round(.2*total));test_count=total-train_count-val_count
    primary={name for name,ids in files.items() if cond[ids[0]]==target}
    source=[name for name in sorted(files) if name not in primary]
    rng=np.random.default_rng(seed+complete.index(target)*1000)
    # Fill the remaining test count with explicitly separate seen-domain controls.
    controls=set();control_groups=set()
    for j in range(test_count-len(primary)):
        c=j%5
        options=[name for name in source if y[files[name][0]]==c and (cond[files[name][0]],c) not in control_groups]
        name=options[int(rng.integers(len(options)))];controls.add(name);control_groups.add((cond[files[name][0]],c))
    grouped=collections.defaultdict(list)
    for name in source:
        if name not in controls:grouped[(int(y[files[name][0]]),cond[files[name][0]])].append(name)
    validation=set();base,remainder=divmod(val_count,7)
    for c in range(7):
        groups=sorted(k for k in grouped if k[0]==c);needed=base+(c<remainder)
        if needed>len(groups):raise ValueError('Insufficient class/condition groups for validation count')
        for j in rng.choice(len(groups),needed,replace=False):
            names=sorted(grouped[groups[int(j)]]);validation.add(names[int(rng.integers(len(names)))])
    training_files=set(source)-controls-validation
    assert len(training_files)==train_count and len(validation)==val_count and len(primary|controls)==test_count
    tr=np.array([r['record_id'] in training_files for r in rows]);va=np.array([r['record_id'] in validation for r in rows]);te=np.array([r['record_id'] in primary|controls for r in rows])
    partition[tr]='train';partition[va]='validation';partition[te]='test'
    role[tr&(y<5)]='base_train';role[tr&(y>=5)]='novel_support_pool'
    role[va&(y<5)]='base_val';role[va&(y>=5)]='unused_novel_validation'
    role[np.array([r['record_id'] in primary for r in rows])]='query'
    role[np.array([r['record_id'] in controls for r in rows])]='seen_test_control'
    training=sorted(set(cond[tr]));records=collections.defaultdict(set);hashes=collections.defaultdict(set)
    for row,p in zip(rows,partition):
        if not p.startswith('excluded'):
            records[row['record_id']].add(p);hashes[row['raw_sha256']].add(p)
    if any(len(s)!=1 for mapping in (records,hashes) for s in mapping.values()):raise ValueError('File/hash partition leakage')
    if target in set(cond[tr|va]):raise ValueError('Unseen target entered fitting/validation')
    counts={p:len({r['record_id'] for r,t in zip(rows,partition) if t==p}) for p in ('train','validation','test')}
    total=sum(counts.values())
    audit={'target':target,'seed':seed,'train_conditions':training,'validation_conditions':sorted(set(cond[va])),
        'test_conditions':sorted(set(cond[te])),'unseen_test_condition':target,'requested_ratio':[.7,.2,.1],
        'actual_record_counts':counts,'actual_record_ratio':{p:n/total for p,n in counts.items()},
        'actual_window_counts':{p:int((partition==p).sum()) for p in counts},'complete_conditions':complete,
        'incomplete_source_conditions_retained':skipped,'unseen_test_files':len(primary),'seen_control_test_files':len(controls),
        'unseen_test_file_ids':sorted(primary),'seen_control_file_ids':sorted(controls),
        'actual_supervised_base_train_windows':int((role=='base_train').sum()),
        'actual_base_validation_windows':int((role=='base_val').sum()),
        'novel_label_policy':'only K per class in source train partition; other novel labels not used to fit any model/scaler/calibrator',
        'scope':'unseen combination; all speed and load levels present in training; specimen/run identities unknown'}
    return role,partition,cond,training,audit


def cross_episode(y,role,cond,conditions,rng):
    ids=[]
    for c in range(5):
        available=[name for name in conditions if np.sum((role=='base_train')&(cond==name)&(y==c))>=7]
        first,second=[available[int(i)] for i in rng.choice(len(available),2,replace=False)]
        s=np.flatnonzero((role=='base_train')&(cond==first)&(y==c));q=np.flatnonzero((role=='base_train')&(cond==second)&(y==c))
        ids.extend(rng.choice(s,5,replace=False));ids.extend(rng.choice(q,7,replace=False))
    support=np.concatenate([np.arange(c*12,c*12+5) for c in range(5)])
    query=np.concatenate([np.arange(c*12+5,(c+1)*12) for c in range(5)])
    return np.array(ids),support,query


def global_validation(z,y,role,cond,conditions):
    tr=np.flatnonzero(role=='base_train');va=np.flatnonzero(role=='base_val');predictions=[]
    yy=torch.as_tensor(y[tr],device=z.device)
    for draw in z:
        means,chol=fit_reference(draw[tr],yy);predictions.append(mahalanobis_logits(draw[va],means,chol).softmax(-1))
    return torch.stack(predictions).mean(0),torch.as_tensor(y[va],device=z.device)


def cross_supports(rows,y,role,target,seed,k,episodes):
    rng=np.random.default_rng(seed*100000+CASES.index(target)*10000+k*100) if target in CASES else np.random.default_rng(seed*100000+sum(map(ord,target))*100+k)
    output=[]
    for episode in range(episodes):
        chosen=[]
        for c in (5,6):
            grouped=collections.defaultdict(list)
            for i in np.flatnonzero((role=='novel_support_pool')&(y==c)):grouped[rows[i]['record_id']].append(int(i))
            names=sorted(grouped)
            if k>len(names):raise ValueError('K exceeds distinct source support recordings')
            chosen.append([int(rng.choice(grouped[names[j]])) for j in rng.choice(len(names),k,replace=False)])
        output.append({'target':target,'seed':seed,'k':k,'episode':episode,'indices':chosen,
            'sample_ids':[[rows[i]['sample_id'] for i in ids] for ids in chosen]})
    return output


def evaluate_cross(z,rows,y,role,target,args,seed,modality,method,result):
    tr=np.flatnonzero(role=='base_train');va=np.flatnonzero(role=='base_val');qu=np.flatnonzero(role=='query')
    all_test=np.flatnonzero((role=='query')|(role=='seen_test_control'));primary_mask=role[all_test]=='query'
    transformed=[];references=[]
    with torch.no_grad():
        for draw in z:
            zz=torch.as_tensor(draw);means,chol=fit_reference(zz[tr],torch.as_tensor(y[tr]),shrinkage=getattr(args,'shrinkage',.2));transformed.append(whiten(zz,chol).numpy());references.append(whiten(means,chol).numpy())
    w=np.stack(transformed).astype(np.float64);base=np.stack(references).astype(np.float64);dimension=z.shape[-1]
    def logits(ids,means):
        values=w[:,ids];squared=(values**2).sum(-1)[:,:,None]+(means**2).sum(-1)[:,None]-2*np.einsum('mnd,mcd->mnc',values,means)
        return -np.maximum(squared,0)/dimension
    # Separate five-known diagnostic, without novel competitors.
    known=qu[y[qu]<5];temperature=fit_temperature(logits(va,base),y[va]);p=softmax(logits(known,base)/temperature,axis=-1).mean(0)
    cm=np.bincount(5*y[known]+p.argmax(-1),minlength=25).reshape(5,5);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    diagnostic={'normalization':'window_zscore','modality':modality,'method':method,'condition':target,'seed':seed,
        'accuracy':float((p.argmax(-1)==y[known]).mean()),'macro_f1':float(f1.mean()),'confusion':cm.tolist(),
        'nll':float(-np.log(p[np.arange(len(known)),y[known]].clip(1e-12)).mean()),'temperature':temperature,'scope':'five known candidates only'}
    evaluations=[];blocks=[];allblocks=[];uq=[];acceptance=[];shots=[];supports=[]
    for k in args.shots:
        for support in cross_supports(rows,y,role,target,seed,k,args.episodes):
            means=np.concatenate([base]+[w[:,ids].mean(1)[:,None] for ids in support['indices']],axis=1)
            vl=logits(va,means);all_logits=logits(all_test,means);ql=all_logits[:,primary_mask];temperature=fit_temperature(vl,y[va])
            v,_,_,vmi=uncertainty(softmax(vl/temperature,axis=-1));p,entropy,expected,mi=uncertainty(softmax(ql/temperature,axis=-1));raw=softmax(ql,axis=-1).mean(0)
            threshold=threshold_confidence(y[va],v,args.target_error);radius=float(np.quantile(-vl.mean(0)[np.arange(len(va)),y[va]],.95));limit=float(np.quantile(vmi,.95))
            pred=p.argmax(-1);accepted=(p.max(-1)>=threshold)&(-ql.mean(0)[np.arange(len(qu)),pred]<=radius)
            if method=='bayesian':accepted&=mi<=limit
            evaluations.append({'normalization':'window_zscore','modality':modality,'method':method,'condition':target,'seed':seed,'k':k,'episode':support['episode'],
                'calibrated':metrics(y[qu],p),'raw':metrics(y[qu],raw),'selective':selective_metrics(pred==y[qu],accepted),
                'overall_test':metrics(y[all_test],softmax(all_logits/temperature,axis=-1).mean(0)),
                'temperature':temperature,'confidence_threshold':threshold,'radius':radius,'mi_limit':limit,'mean_mi':float(mi.mean())})
            blocks.append(p);allblocks.append(softmax(all_logits/temperature,axis=-1).mean(0));uq.append(np.column_stack([entropy,expected,mi]));acceptance.append(accepted);shots.append(k);supports.append(support['indices'][0]+support['indices'][1])
    filename=f'window_zscore_{modality}_{method}_{target}_s{seed}'
    save_predictions(result/'predictions'/f'{filename}.npz',probabilities=np.stack(blocks),uncertainties=np.stack(uq),acceptance=np.stack(acceptance),shots=np.array(shots),true_labels=y[qu],query_indices=qu,
        all_test_probabilities=np.stack(allblocks),all_test_indices=all_test,all_test_labels=y[all_test])
    dump(result/'predictions'/f'{filename}_supports.json',supports)
    return evaluations,diagnostic


def summarize_cross(result,evaluations,diagnostics):
    summary=summarize(result,evaluations)
    # Retain the numerical table while replacing the same-condition explanation.
    table=(result/'EXPERIMENT_REPORT.md').read_text(encoding='utf-8').split('|输入处理|',1)[1]
    config=json.loads((result/'config.json').read_text(encoding='utf-8'))
    lines=['# '+TITLE,'','指定未见转速—载荷组合不出现在训练或验证。训练保留每个单独转速/载荷档。目标组合事先固定为'+', '.join(config['cases'])+'。',
        '','## 按CSV数量划分','','用户指定按文件数量70/20/10，326份非重复CSV按四舍五入分为228训练、65验证、33测试（69.94/19.94/10.12%）。同一CSV全部窗口保持同一角色。25Hz/75%中无矛盾的部分标签文件保留作源数据，其缺失类别不补造。',
        '','指定未见目标组合包含27或28份CSV，全部作为跨工况测试；测试集余5或6份CSV为已见工况参照。两部分分别评价，不能把全部33份混合分数冒充严格跨工况分数。',
        '','|目标组合|种子|训练文件|验证文件|测试文件|未见工况文件|已见参照文件|实际文件比例（训练/验证/测试）|','|---|---:|---:|---:|---:|---:|---:|---|']
    for file in sorted((result/'splits').glob('*.json')):
        s=json.loads(file.read_text(encoding='utf-8'));c=s['actual_record_counts'];r=s['actual_record_ratio']
        lines.append(f"|{s['target']}|{s['seed']}|{c['train']}|{c['validation']}|{c['test']}|{s['unseen_test_files']}|{s['seen_control_test_files']}|{r['train']:.2%}/{r['validation']:.2%}/{r['test']:.2%}|")
    lines.extend(['','原始分区包含七类，但模型只用源训练五基类标签。两复合新类仅从源训练分区各抽K=1/5/10个窗口，来自K个不同CSV；不是K个独立轴承。源验证复合标签封存，测试标签只评价。',
        '','## 训练与评价','','逐窗逐通道z-score、图像及头标准化统计仅拟合基类训练。CNN/头均用共享收缩全协方差马氏平方距离/维数，收缩0.2。训练episode的支持与查询来自两个不同源训练组合；验证/测试的基类原型及协方差由全部源训练五基类拟合。',
        '','30轮编码器×15步，25轮头×10步；标签平滑0.05。BNN仅头，MC训练2/评价20，KL/P系数0.01，属于带权变分判别代理。温度与确认阈值只用源验证基类，不给新组合或复合类提供5%风险保证。',
        '','## 未见工况子集GFSL结果','','20组支持抽样先按目标组合/种子平均，再描述性汇总任务/种子；不能当独立轴承重复或总体置信区间。以下表格及主要图窗只用未见工况测试子集。',
        '','|输入处理|'+table])
    overall=[]
    for key in sorted(set((r['modality'],r['method'],r['k']) for r in evaluations)):
        selected=[r for r in evaluations if (r['modality'],r['method'],r['k'])==key]
        overall.append({**dict(zip(('modality','method','k'),key)),**{name:float(np.mean([r['overall_test'][name] for r in selected])) for name in ('accuracy','macro_f1','novel_f1','ece')},'scope':'all 33 test files; mixed unseen and seen controls'})
    dump(result/'overall_test_summary.json',overall)
    lines.extend(['','## 全部33份测试文件（混合域）','','以下分数包含已见参照，不能等同严格跨工况结果。','', '|输入|头|K|准确率|Macro-F1|复合F1|ECE|','|---|---|---:|---:|---:|---:|---:|'])
    for r in overall:lines.append(f"|{r['modality']}|{r['method']}|{r['k']}|{r['accuracy']:.2%}|{r['macro_f1']:.2%}|{r['novel_f1']:.2%}|{r['ece']:.4f}|")
    if diagnostics:
        groups=collections.defaultdict(list)
        for d in diagnostics:groups[(d['modality'],d['method'])].append(d)
        ds=[{**dict(zip(('modality','method'),key)),**{name:float(np.mean([r[name] for r in v])) for name in ('accuracy','macro_f1','nll')},'task_seed_count':len(v)} for key,v in sorted(groups.items())]
        dump(result/'known_only_summary.json',ds)
        lines.extend(['','## 五基类候选诊断','','暂不让复合类参与排序，仅检验基类跨工况表征。','', '|输入|头|准确率|Macro-F1|NLL|','|---|---|---:|---:|---:|'])
        for d in ds:lines.append(f"|{d['modality']}|{d['method']}|{d['accuracy']:.2%}|{d['macro_f1']:.2%}|{d['nll']:.4f}|")
    lines.extend(['','## 边界','','LASPI为预制机械缺陷，实体试件及运行独立性未知。本轮无电蚀机理、物理损失或RUL，3个目标组合不能代表全部未知工况。原同工况结果及跨工况首版保留。图窗中文，不自动导出图片。'])
    (result/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run(args):
    torch.set_num_threads(2);start=time.perf_counter();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result=Path(args.output or ROOT/'results'/'cross_csv702010_20261012').resolve()
    if result.exists() and not args.resume:
        original=result;i=2
        while result.exists():result=original.with_name(original.name+f'_run{i}');i+=1
    for name in ('predictions','splits','source_snapshot'):(result/name).mkdir(parents=True,exist_ok=True)
    sources=('run_laspi_cross_702010.py','run_laspi_same_condition.py','mahalanobis_metric.py','bayesian_layers.py','prepare_laspi.py','run_laspi_gfsl.py')
    for source in sources:
        if not (result/'source_snapshot'/source).exists():shutil.copy2(ROOT/source,result/'source_snapshot'/source)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','no_show')};config['output']=str(result)
    config.update({'protocol':'cross_condition_702010','title':TITLE,'normalizations':['window_zscore'],'requested_ratio':[.7,.2,.1],'split_quantity_unit':'CSV','actual_record_counts':[228,65,33],
        'metric':'shared shrinkage full covariance Mahalanobis squared / dimension','shrinkage':.2,'train_episode':'each class uses different source support/query conditions, including valid partial-condition labels',
        'prototype_covariance_fit':'all five source training classes only','novel_support_domain':'source training conditions; K distinct CSV files per class',
        'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'torch':torch.__version__,'numpy':np.__version__,
        'physics':False,'calibration':'source base validation only','code_hashes':{s:sha256(ROOT/s) for s in sources}})
    config['display_scope']='unseen target condition subset; overall 33-file test scores reported separately'
    if args.resume:
        old=json.loads((result/'config.json').read_text(encoding='utf-8'))
        if any(old.get(k)!=v for k,v in config.items() if k not in ('code_hashes','cache_mode')):raise ValueError('Resume scientific configuration changed')
        snapshot=result/'resume_source_snapshot_20261012'
        snapshot.mkdir(exist_ok=True)
        for source in sources:shutil.copy2(ROOT/source,snapshot/source)
        dump(result/'resume_runtime.json',{'cache_mode':args.cache_mode,'scientific_parameters_unchanged':True,'source_hashes':config['code_hashes']})
    else:dump(result/'config.json',config)
    metrics_path=result/'metrics.json';diag_path=result/'known_only_metrics.json'
    evaluations=json.loads(metrics_path.read_text(encoding='utf-8')) if args.resume and metrics_path.exists() else []
    diagnostics=json.loads(diag_path.read_text(encoding='utf-8')) if args.resume and diag_path.exists() else []
    print('正在读取归一化缓存：'+('一次读入内存，避免训练期间访问H盘内存映射。' if args.cache_mode=='ram' else '磁盘内存映射。'),flush=True)
    rows,y,images=load_samples(ROOT/'prepared_window_zscore',mmap=args.cache_mode=='mmap')
    print(f'缓存已就绪：{images.nbytes/2**30:.2f} GiB；结果目录：{result}',flush=True)
    total_groups=len(args.cases)*len(args.seeds)*len(args.modalities)*2
    expected_evaluations=total_groups*len(args.shots)*args.episodes
    for target in args.cases:
        for seed in args.seeds:
            role,partition,cond,training,split=split_cross(rows,target,seed)
            filename=f'{target}_s{seed}';dump(result/'splits'/f'{filename}.json',split)
            manifest=[{'sample_id':r['sample_id'],'record_id':r['record_id'],'sha256':r['raw_sha256'],'condition':cond[i],'label':int(y[i]),'partition':partition[i],'role':role[i]} for i,r in enumerate(rows)]
            splitfile=result/'splits'/f'{filename}.csv'
            if splitfile.exists():
                with splitfile.open(encoding='utf-8-sig') as f:previous=list(csv.DictReader(f))
                if [(r['sample_id'],r['role']) for r in previous]!=[(r['sample_id'],r['role']) for r in manifest]:raise ValueError('Resume split changed')
            else:csv_out(splitfile,manifest)
            scaler=image_scaler(images,role=='base_train')
            for modality in args.modalities:
                folder=result/'models'/f'{modality}_{target}_s{seed}';folder.mkdir(parents=True,exist_ok=True);dump(folder/'image_scaler.json',scaler)
                features,history=train_encoder(images,rows,y,role,cond,training,MODALITIES[modality],scaler,args,seed,folder,device,cross_episode,global_validation)
                for method in ('deterministic','bayesian'):
                    if any((m['modality'],m['condition'],m['seed'],m['method'])==(modality,target,seed,method) for m in evaluations):continue
                    z,hh=train_head(features,y,role,cond,training,args,seed,folder/method,method=='bayesian',device,cross_episode,global_validation)
                    new,diagnostic=evaluate_cross(z,rows,y,role,target,args,seed,modality,method,result)
                    evaluations.extend(new);diagnostics.append(diagnostic);dump(metrics_path,evaluations);dump(diag_path,diagnostics);summarize_cross(result,evaluations,diagnostics)
                    completed=len({(r['condition'],r['seed'],r['modality'],r['method']) for r in evaluations})
                    print(f'COMPLETE {target} {modality} {method} seed={seed} | 总进度 {completed}/{total_groups}组；全部完成后弹出图窗。',flush=True)
                    dump(result/'progress.json',{'finished':False,'completed_groups':completed,'total_groups':total_groups,'evaluations':len(evaluations),'expected_evaluations':expected_evaluations})
    summary=summarize_cross(result,evaluations,diagnostics);dump(result/'execution.json',{'finished':True,'elapsed_seconds':time.perf_counter()-start,'evaluations':len(evaluations),'gpu':config['gpu']})
    dump(result/'progress.json',{'finished':True,'completed_groups':total_groups,'total_groups':total_groups,'evaluations':len(evaluations),'expected_evaluations':expected_evaluations})
    print('RESULT',result,flush=True)
    for record in summary:print('SUMMARY',record,flush=True)
    if not args.no_show:
        print('实验全部完成，正在打开中文结果图窗；窗口关闭后程序结束。',flush=True)
        from show_laspi_same_condition import show
        show(result)
    return result


def arguments():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);ap.add_argument('--resume',action='store_true');ap.add_argument('--no-show',action='store_true')
    ap.add_argument('--cache-mode',choices=('ram','mmap'),default='ram')
    ap.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--modalities',nargs='+',choices=tuple(MODALITIES),default=['vibration','multisignal'])
    ap.add_argument('--shots',nargs='+',type=int,default=[1,5,10]);ap.add_argument('--episodes',type=int,default=20)
    ap.add_argument('--encoder-epochs',type=int,default=30);ap.add_argument('--encoder-steps',type=int,default=15);ap.add_argument('--head-epochs',type=int,default=25);ap.add_argument('--head-steps',type=int,default=10)
    ap.add_argument('--mc-train',type=int,default=2);ap.add_argument('--mc-eval',type=int,default=20);ap.add_argument('--kl-beta',type=float,default=.01);ap.add_argument('--prior-std',type=float,default=1.);ap.add_argument('--target-error',type=float,default=.05)
    args=ap.parse_args()
    if min(args.shots)<1 or max(args.shots)>32:raise ValueError('Source supports require 1..32 distinct recordings')
    if min(args.episodes,args.encoder_epochs,args.encoder_steps,args.head_epochs,args.head_steps,args.mc_train,args.mc_eval)<=0:raise ValueError('Positive budgets required')
    return args


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    run(arguments())
