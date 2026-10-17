"""LASPI file-held same-condition GFSL: normalized inputs and Mahalanobis only.

Diagnostic experiment, not independent bearing validation or cross-condition proof.
"""
from __future__ import annotations
import argparse,csv,json,math,os,shutil,time,collections
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax
from run_laspi_gfsl import (Encoder,seed_all,load_samples,image_scaler,batch,encode,
    save_checkpoint,save_predictions,fit_temperature,metrics,threshold_confidence,
    selective_metrics,MODALITIES)
from prepare_laspi import ROOT,DEFAULT_SOURCE,prepare,dump,csv_out,sha256,CN
from bayesian_layers import FeatureHead,uncertainty
from mahalanobis_metric import fit_reference,whiten,mahalanobis_logits


def save_feature_array(path,values):
    path=Path(path);temporary=path.with_name(path.name+'.writing.tmp')
    with temporary.open('wb') as stream:
        np.save(stream,values);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

TITLE='同工况齿轮—轴承复合故障：归一化与统一马氏度量对照'


def split_files(rows,seed):
    """Same settings in all roles; a whole CSV can only have one role."""
    by_group=collections.defaultdict(dict)
    for i,r in enumerate(rows):
        if not r['excluded_duplicate']:
            by_group[(r['speed_setting_hz'],r['load_percent'],r['label'])].setdefault(r['record_id'],[]).append(i)
    conditions=[];skipped=[]
    for speed in (25,35,45):
        for load in (0,25,50,75):
            counts=[len(by_group[(speed,load,c)]) for c in range(7)]
            if min(counts[:5])>=3 and min(counts[5:])>=2:conditions.append(f'{speed}hz_{load}pct')
            else:skipped.append({'speed':speed,'load':load,'file_counts':counts})
    role=np.full(len(rows),'excluded_incomplete_condition',dtype=object)
    role[np.array([r['excluded_duplicate'] for r in rows])]='excluded_duplicate'
    cond=np.array([f"{r['speed_setting_hz']}hz_{r['load_percent']}pct" for r in rows])
    rng=np.random.default_rng(seed)
    for name in conditions:
        speed,load=map(int,name.replace('hz_',' ').replace('pct','').split())
        for c in range(7):
            group=by_group[(speed,load,c)];names=sorted(group);names=[names[j] for j in rng.permutation(len(names))]
            assignments=([('base_train',n) for n in names[:-2]]+[('base_val',names[-2]),('query',names[-1])]) if c<5 else ([('novel_support_pool',names[0])]+[('query',n) for n in names[1:]])
            for label,record in assignments:role[group[record]]=label
    roles=collections.defaultdict(set);hashes=collections.defaultdict(set)
    for r,label in zip(rows,role):
        if not label.startswith('excluded'):
            roles[r['record_id']].add(label);hashes[r['raw_sha256']].add(label)
    if any(len(x)>1 for x in roles.values()) or any(len(x)>1 for x in hashes.values()):raise ValueError('File/hash leakage')
    return role,cond,conditions,skipped


def episode_ids(y,role,cond,conditions,rng):
    name=conditions[int(rng.integers(len(conditions)))];ids=[]
    for c in range(5):
        pool=np.flatnonzero((role=='base_train')&(cond==name)&(y==c))
        ids.extend(rng.choice(pool,12,replace=False))
    support=np.concatenate([np.arange(c*12,c*12+5) for c in range(5)])
    query=np.concatenate([np.arange(c*12+5,(c+1)*12) for c in range(5)])
    return np.array(ids),support,query


def validation_probabilities(z,y,role,cond,conditions):
    probabilities=[];labels=[]
    for name in conditions:
        train=np.flatnonzero((role=='base_train')&(cond==name));val=np.flatnonzero((role=='base_val')&(cond==name))
        yy=torch.as_tensor(y[train],device=z.device)
        predictions=[]
        for draw in z:
            means,chol=fit_reference(draw[train],yy)
            predictions.append(mahalanobis_logits(draw[val],means,chol).softmax(-1))
        probabilities.append(torch.stack(predictions).mean(0));labels.extend(y[val])
    return torch.cat(probabilities),torch.as_tensor(labels,device=z.device)


def train_encoder(images,rows,y,role,cond,conditions,channels,scaler,args,seed,path,device,episode_fn=None,validation_fn=None):
    episode_fn=episode_fn or episode_ids;validation_fn=validation_fn or validation_probabilities
    seed_all(seed);model=Encoder(len(channels)).to(device);model.classifier=nn.Identity()
    if (path/'encoder_features.npy').exists():
        return np.load(path/'encoder_features.npy'),json.loads((path/'encoder_history.json').read_text(encoding='utf-8'))
    if (path/'best_encoder.pt').exists() and (path/'encoder_history.json').exists():
        print(f'恢复 {path.name}：使用已保存的最佳编码器补提取特征，无需重新训练。',flush=True)
        model.load_state_dict(torch.load(path/'best_encoder.pt',weights_only=True,map_location=device))
        features=encode(model,images,np.arange(len(rows)),channels,scaler,device)
        save_feature_array(path/'encoder_features.npy',features)
        return features,json.loads((path/'encoder_history.json').read_text(encoding='utf-8'))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(seed);history=[];best=math.inf;state=None
    selected=np.flatnonzero((role=='base_train')|(role=='base_val'))
    for epoch in range(1,args.encoder_epochs+1):
        model.train();losses=[]
        for _ in range(args.encoder_steps):
            ids,support,query=episode_fn(y,role,cond,conditions,rng);yy=torch.as_tensor(y[ids],device=device)
            optimizer.zero_grad(set_to_none=True);z=model(batch(images,ids,channels,scaler,device))
            means,chol=fit_reference(z[support],yy[support]);loss=F.cross_entropy(mahalanobis_logits(z[query],means,chol),yy[query],label_smoothing=.05)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        if epoch==1 or epoch%2==0 or epoch==args.encoder_epochs:
            values=encode(model,images,selected,channels,scaler,device)
            with torch.no_grad():
                p,t=validation_fn(torch.as_tensor(values,device=device)[None],y[selected],role[selected],cond[selected],conditions)
                nll=float(-p[torch.arange(len(t),device=device),t].clamp_min(1e-12).log().mean());acc=float((p.argmax(-1)==t).float().mean())
            history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'base_val_nll':nll,'base_val_accuracy':acc})
            if nll<best:best=nll;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%10==0:print(f'ENCODER {path.name} round={epoch} valAcc={acc:.4f} valNLL={nll:.4f}',flush=True)
    path.mkdir(parents=True,exist_ok=True);save_checkpoint(state,path/'best_encoder.pt');save_checkpoint(model.state_dict(),path/'last_encoder.pt');dump(path/'encoder_history.json',history)
    model.load_state_dict(state);features=encode(model,images,np.arange(len(rows)),channels,scaler,device)
    save_feature_array(path/'encoder_features.npy',features);return features,history


def train_head(features,y,role,cond,conditions,args,seed,path,bayesian,device,episode_fn=None,validation_fn=None):
    episode_fn=episode_fn or episode_ids;validation_fn=validation_fn or validation_probabilities
    if (path/'posterior_features.npy').exists():return np.load(path/'posterior_features.npy'),json.loads((path/'head_history.json').read_text(encoding='utf-8'))
    seed_all(seed);train=role=='base_train';mu=features[train].mean(0);sd=features[train].std(0).clip(1e-3)
    x=torch.as_tensor((features-mu)/sd,device=device);model=FeatureHead(bayesian,args.prior_std).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.002);rng=np.random.default_rng(seed);history=[];best=math.inf;state=None
    selected=np.flatnonzero((role=='base_train')|(role=='base_val'))
    for epoch in range(1,args.head_epochs+1):
        losses=[]
        for _ in range(args.head_steps):
            ids,support,query=episode_fn(y,role,cond,conditions,rng);yy=torch.as_tensor(y[ids],device=device)
            optimizer.zero_grad(set_to_none=True);total=0.;draws=args.mc_train if bayesian else 1
            for _ in range(draws):
                model.resample();z=model(x[ids]);means,chol=fit_reference(z[support],yy[support])
                total+=F.cross_entropy(mahalanobis_logits(z[query],means,chol),yy[query],label_smoothing=.05)
            total=total/draws+args.kl_beta*model.kl()/model.count();total.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(total.detach()))
        with torch.no_grad():
            valdraws=[]
            for _ in range(5 if bayesian else 1):model.resample();valdraws.append(model(x[selected]))
            p,t=validation_fn(torch.stack(valdraws),y[selected],role[selected],cond[selected],conditions)
            nll=float(-p[torch.arange(len(t),device=device),t].clamp_min(1e-12).log().mean());acc=float((p.argmax(-1)==t).float().mean())
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'base_val_nll':nll,'base_val_accuracy':acc,'kl_per_parameter':float(model.kl().detach()/model.count())})
        if nll<best:best=nll;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    path.mkdir(parents=True,exist_ok=True);save_checkpoint(state,path/'best_head.pt');save_checkpoint(model.state_dict(),path/'last_head.pt');dump(path/'head_history.json',history)
    dump(path/'feature_scaler.json',{'mean':mu.tolist(),'std':sd.tolist(),'fit_indices':np.flatnonzero(train).tolist()})
    model.load_state_dict(state);features=[];weights=[]
    with torch.no_grad():
        for _ in range(args.mc_eval if bayesian else 1):
            model.resample();features.append(model(x).cpu().numpy())
            weights.append({f'{i}_{k}':getattr(layer,k+'_sample').detach().cpu() for i,layer in enumerate(model.layers) for k in ('weight','bias')})
    z=np.stack(features);save_checkpoint(weights,path/'posterior_weight_samples.pt');save_feature_array(path/'posterior_features.npy',z);return z,history


def support_sets(rows,y,role,cond,name,seed,k,episodes):
    # Same split, same exact supports across modalities, normalization and heads.
    speed,load=map(int,name.replace('hz_',' ').replace('pct','').split());rng=np.random.default_rng(seed*100000+speed*1000+load*10+k)
    output=[]
    for episode in range(episodes):
        ids=[rng.choice(np.flatnonzero((role=='novel_support_pool')&(cond==name)&(y==c)),k,replace=False) for c in (5,6)]
        output.append({'condition':name,'seed':seed,'k':k,'episode':episode,'indices':[i.tolist() for i in ids],
            'sample_ids':[[rows[j]['sample_id'] for j in i] for i in ids]})
    return output


def evaluate(z,rows,y,role,cond,conditions,args,seed,normalization,modality,method,result):
    evaluations=[]
    for name in conditions:
        ti=np.flatnonzero((role=='base_train')&(cond==name));vi=np.flatnonzero((role=='base_val')&(cond==name));qi=np.flatnonzero((role=='query')&(cond==name))
        whitened=[];base_means=[]
        # Identical weight draw transforms train/support/val/query; one shared covariance per draw.
        with torch.no_grad():
            for draw in z:
                zz=torch.as_tensor(draw);means,chol=fit_reference(zz[ti],torch.as_tensor(y[ti]));whitened.append(whiten(zz,chol).numpy());base_means.append(whiten(means,chol).numpy())
        w=np.stack(whitened).astype(np.float64);base=np.stack(base_means).astype(np.float64);dimension=z.shape[-1]
        blocks=[];uqblocks=[];acceptance=[];shot_array=[];support_array=[]
        for k in args.shots:
            for support in support_sets(rows,y,role,cond,name,seed,k,args.episodes):
                means=np.concatenate([base]+[w[:,ids].mean(1)[:,None] for ids in support['indices']],axis=1)
                def logits(ids):
                    values=w[:,ids];d=(values*values).sum(-1)[:,:,None]+(means*means).sum(-1)[:,None]-2*np.einsum('mnd,mcd->mnc',values,means)
                    return -np.maximum(d,0)/dimension
                vl=logits(vi);ql=logits(qi);temperature=fit_temperature(vl,y[vi])
                vp,_,_,vmi=uncertainty(softmax(vl/temperature,axis=-1));p,entropy,expected,mi=uncertainty(softmax(ql/temperature,axis=-1))
                raw=softmax(ql,axis=-1).mean(0);threshold=threshold_confidence(y[vi],vp,args.target_error)
                radius=float(np.quantile(-vl.mean(0)[np.arange(len(vi)),y[vi]],.95));limit=float(np.quantile(vmi,.95))
                predicted=p.argmax(-1);accepted=(p.max(-1)>=threshold)&(-ql.mean(0)[np.arange(len(qi)),predicted]<=radius)
                if method=='bayesian':accepted&=mi<=limit
                record={'normalization':normalization,'modality':modality,'method':method,'condition':name,'seed':seed,'k':k,'episode':support['episode'],
                    'calibrated':metrics(y[qi],p),'raw':metrics(y[qi],raw),'selective':selective_metrics(predicted==y[qi],accepted),
                    'temperature':temperature,'confidence_threshold':threshold,'radius':radius,'mi_limit':limit,'mean_mi':float(mi.mean())}
                evaluations.append(record);blocks.append(p);uqblocks.append(np.column_stack([entropy,expected,mi]));acceptance.append(accepted);shot_array.append(k);support_array.append(np.r_[support['indices'][0],support['indices'][1]].tolist())
        # Ragged support lengths have a JSON sidecar, no pickle arrays.
        filename=f'{normalization}_{modality}_{method}_{name}_s{seed}'
        save_predictions(result/'predictions'/f'{filename}.npz',probabilities=np.stack(blocks),uncertainties=np.stack(uqblocks),
            acceptance=np.stack(acceptance),shots=np.array(shot_array),true_labels=y[qi],query_indices=qi)
        dump(result/'predictions'/f'{filename}_supports.json',support_array)
    return evaluations


def summarize(result,evaluations):
    grouped=collections.defaultdict(list)
    for r in evaluations:grouped[(r['normalization'],r['modality'],r['method'],r['condition'],r['seed'],r['k'])].append(r)
    rows=[]
    for key,items in grouped.items():
        rec=dict(zip(('normalization','modality','method','condition','seed','k'),key))
        for name in ('accuracy','macro_f1','base_f1','novel_f1','nll','ece','brier'):rec[name]=float(np.mean([i['calibrated'][name] for i in items]))
        rec['raw_ece']=float(np.mean([i['raw']['ece'] for i in items]));rec['coverage']=float(np.mean([i['selective']['coverage'] for i in items]))
        errors=[i['selective']['selective_error'] for i in items if i['selective']['selective_error'] is not None]
        rec['selective_error']=float(np.mean(errors)) if errors else None;rows.append(rec)
    csv_out(result/'task_seed_summary.csv',rows);summary=[]
    for key in sorted(set((r['normalization'],r['modality'],r['method'],r['k']) for r in rows)):
        selected=[r for r in rows if (r['normalization'],r['modality'],r['method'],r['k'])==key]
        rec=dict(zip(('normalization','modality','method','k'),key))
        for name in ('accuracy','macro_f1','base_f1','novel_f1','nll','ece','raw_ece','coverage','selective_error'):
            values=[r[name] for r in selected if r[name] is not None];rec[name]=float(np.mean(values)) if values else None
        rec['conditions_seeds']=len(selected);summary.append(rec)
    dump(result/'summary.json',summary)
    lines=['# '+TITLE,'','同工况按CSV隔离的定位实验；不同文件是否独立运行、试件ID均未知。不能当跨独立轴承证明。',
        '','五个基类每条件2/1/1文件（健康剩3文件时1/1/1）；两复合类各1支持文件，其余查询。K窗来自同一支持文件，支持/查询文件不重叠。',
        '','编码器见全部11种设置的基类训练文件，episode按同一条件构造；复合类只用K标签。训练及分类同用共享全协方差收缩马氏平方距离，收缩0.2、按维数缩放，训练协方差不回传梯度。',
        '','逐窗逐通道z-score移除幅值/RMS信息；与原去均值对照分别训练，不能预设一定改善。后续图像及头输入统计只拟合基类训练集。',
        '','BNN仅特征头，Gaussian后验、MC训练2/评价20，KL/P系数0.01；校准仅基类验证，不保证复合风险或跨工况迁移。',
        '','支持抽样先在工况/种子内平均，再对工况/种子描述性平均，不能当独立轴承重复。','',
        '|输入处理|输入|头|K|Macro-F1|复合F1|ECE|确认覆盖|确认错误|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        err='无确认' if s['selective_error'] is None else f"{s['selective_error']:.2%}"
        lines.append(f"|{s['normalization']}|{s['modality']}|{s['method']}|{s['k']}|{s['macro_f1']:.2%}|{s['novel_f1']:.2%}|{s['ece']:.4f}|{s['coverage']:.2%}|{err}|")
    (result/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary


def run(args):
    torch.set_num_threads(2);start=time.perf_counter();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result=Path(args.output).resolve() if args.output else ROOT/'results'/'same_condition_20261012'
    if result.exists() and not args.resume:
        base=result;number=2
        while result.exists():result=base.with_name(base.name+f'_run{number}');number+=1
    result.mkdir(parents=True,exist_ok=True);(result/'predictions').mkdir(exist_ok=True);(result/'source_snapshot').mkdir(exist_ok=True)
    for filename in ('run_laspi_same_condition.py','prepare_laspi.py','mahalanobis_metric.py','bayesian_layers.py','run_laspi_gfsl.py'):
        if not (result/'source_snapshot'/filename).exists():shutil.copy2(ROOT/filename,result/'source_snapshot'/filename)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','no_show')}
    config['output']=str(result)
    config.update({'title':TITLE,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'metric':'shared shrinkage full covariance Mahalanobis squared / dimension, same implementation train/test',
        'shrinkage':.2,'covariance_gradient':False,'physics':False,'split_unit':'CSV; independent runs/specimens unknown',
        'support_unit':'K nonoverlapping windows from one novel recording per class/condition','calibration':'base validation only',
        'normalizer_fit':'base_train only; per-window z-score is a local transform','code_hashes':{f:sha256(ROOT/f) for f in ('run_laspi_same_condition.py','prepare_laspi.py','mahalanobis_metric.py')}})
    if args.resume:
        old=json.loads((result/'config.json').read_text(encoding='utf-8'));skip={'code_hashes'}
        if any(old.get(k)!=v for k,v in config.items() if k not in skip):raise ValueError('Resume scientific configuration changed')
    else:dump(result/'config.json',config)
    allmetrics=json.loads((result/'metrics.json').read_text(encoding='utf-8')) if args.resume and (result/'metrics.json').exists() else []
    for normalization in args.normalizations:
        cache=ROOT/('prepared' if normalization=='none' else 'prepared_window_zscore')
        prepare(argparse.Namespace(source=DEFAULT_SOURCE,output=cache,normalization=normalization))
        samples,y,images=load_samples(cache)
        for seed in args.seeds:
            role,cond,conditions,skipped=split_files(samples,seed)
            if args.conditions:conditions=[c for c in conditions if c in args.conditions];role[(~np.isin(cond,conditions))&(~np.char.startswith(role.astype(str),'excluded'))]='excluded_not_selected'
            if not conditions:raise ValueError('No complete selected conditions')
            split=[{'sample_id':r['sample_id'],'record_id':r['record_id'],'sha256':r['raw_sha256'],'label':int(y[i]),'condition':cond[i],'role':role[i]} for i,r in enumerate(samples)]
            filename=result/f'split_seed{seed}.csv'
            if filename.exists():
                with filename.open(encoding='utf-8-sig') as f:prior=list(csv.DictReader(f))
                if [(r['sample_id'],r['role']) for r in prior]!=[(r['sample_id'],r['role']) for r in split]:raise ValueError('Normalization comparison split changed')
            else:csv_out(filename,split)
            dump(result/'skipped_conditions.json',skipped)
            scaler=image_scaler(images,role=='base_train')
            for modality in args.modalities:
                destination=result/'models'/f'{normalization}_{modality}_s{seed}';destination.mkdir(parents=True,exist_ok=True)
                dump(destination/'image_scaler.json',scaler)
                features,history=train_encoder(images,samples,y,role,cond,conditions,MODALITIES[modality],scaler,args,seed,destination,device)
                for method in ('deterministic','bayesian'):
                    if any((r['normalization'],r['modality'],r['seed'],r['method'])==(normalization,modality,seed,method) for r in allmetrics):continue
                    z,hh=train_head(features,y,role,cond,conditions,args,seed,destination/method,method=='bayesian',device)
                    allmetrics.extend(evaluate(z,samples,y,role,cond,conditions,args,seed,normalization,modality,method,result))
                    dump(result/'metrics.json',allmetrics);summarize(result,allmetrics)
                    print(f'COMPLETE {normalization} {modality} {method} seed={seed}',flush=True)
    summary=summarize(result,allmetrics);dump(result/'execution.json',{'finished':True,'elapsed_seconds':time.perf_counter()-start,'evaluations':len(allmetrics),'gpu':config['gpu']})
    print('RESULT',result,flush=True)
    for s in summary:
        if s['k']==5:print('5SHOT',s,flush=True)
    from analyze_same_condition_known import known_only
    known_only(result)
    if not args.no_show:
        from show_laspi_same_condition import show
        show(result)
    return result


def arguments():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);ap.add_argument('--resume',action='store_true');ap.add_argument('--no-show',action='store_true')
    ap.add_argument('--normalizations',nargs='+',choices=('none','window_zscore'),default=['none','window_zscore'])
    ap.add_argument('--modalities',nargs='+',choices=tuple(MODALITIES),default=['vibration','multisignal']);ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    ap.add_argument('--conditions',nargs='+');ap.add_argument('--shots',nargs='+',type=int,default=[1,5,10]);ap.add_argument('--episodes',type=int,default=20)
    ap.add_argument('--encoder-epochs',type=int,default=30);ap.add_argument('--encoder-steps',type=int,default=15)
    ap.add_argument('--head-epochs',type=int,default=25);ap.add_argument('--head-steps',type=int,default=10)
    ap.add_argument('--mc-train',type=int,default=2);ap.add_argument('--mc-eval',type=int,default=20);ap.add_argument('--kl-beta',type=float,default=.01);ap.add_argument('--prior-std',type=float,default=1.)
    ap.add_argument('--target-error',type=float,default=.05);args=ap.parse_args()
    if min(args.episodes,args.encoder_epochs,args.encoder_steps,args.head_epochs,args.head_steps,args.mc_train,args.mc_eval)<=0:raise ValueError('Positive budgets required')
    if min(args.shots)<1 or max(args.shots)>31:raise ValueError('Support budget must be 1..31 windows')
    return args


if __name__=='__main__':run(arguments())
