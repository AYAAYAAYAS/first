"""LASPI cross-condition GFSL diagnosis/calibration/abstention; physics OFF.

Whole state-condition folders (all four CSVs) stay in one role. Encoders/head
training use only five source known classes; two composites get exactly K
source-condition windows each. The held-out condition never calibrates.
"""
from __future__ import annotations
import argparse
import csv
import collections
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import rankdata
from prepare_laspi import ROOT,CLASSES,CN,dump,csv_out,sha256,prepare,DEFAULT_SOURCE
from bayesian_layers import FeatureHead,uncertainty

TITLE='跨工况齿轮—轴承复合故障的多信号贝叶斯广义少样本诊断与可靠性确认'
SCENARIOS=('speed_25','speed_35','speed_45','load_0','load_25','load_50','load_75')
METHODS=('euclidean','mahalanobis','bayesian_mahalanobis')
NAMES={'euclidean':'确定性欧氏原型','mahalanobis':'确定性收缩马氏','bayesian_mahalanobis':'贝叶斯收缩马氏'}
MODALITIES={'vibration':[3],'multisignal':list(range(7))}


def save_checkpoint(state,path):
    """Python file stream + verified atomic commit; retain failed temp writes."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);failures=[]
    for attempt in range(1,4):
        temporary=path.with_name(path.name+f'.attempt{attempt}.tmp')
        try:
            with temporary.open('wb') as stream:
                torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
            torch.load(temporary,weights_only=True,map_location='cpu')
            temporary.replace(path);return
        except (OSError,RuntimeError) as error:
            failures.append(str(error));path.parent.mkdir(parents=True,exist_ok=True)
    raise RuntimeError('Checkpoint stream writes failed: '+repr(failures))


def save_predictions(path,**arrays):
    path=Path(path);temporary=path.with_name(path.name+'.writing.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    with np.load(temporary) as saved:
        if set(saved.files)!=set(arrays):raise ValueError('Prediction bundle did not commit completely')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def load_samples(cache,mmap=True,metadata_only=False):
    with (cache/'samples.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    with (cache/'records.csv').open(encoding='utf-8-sig',newline='') as stream:records=list(csv.DictReader(stream))
    counts=collections.Counter(r['sha256'] for r in records)
    hashes={r['record_id']:r['sha256'] for r in records}
    for row in rows:
        for key in ('label','speed_setting_hz','load_percent','window_index','cache_index'):row[key]=int(row[key])
        for key in ('motor_rpm_filename','start_s','duration_s'):row[key]=float(row[key])
        row['raw_sha256']=hashes[row['record_id']]
        row['excluded_duplicate']=counts[row['raw_sha256']]>1
    if metadata_only:images=None
    elif mmap:images=np.load(cache/'stft.npy',mmap_mode='r')
    else:
        # Ordinary stream I/O raises a Python error instead of accessing H-drive mapped pages during training.
        with (cache/'stft.npy').open('rb') as stream:images=np.load(stream,allow_pickle=False)
    return rows,np.array([r['label'] for r in rows]),images


def masks(rows,scenario):
    axis,held=scenario.split('_');held=int(held)
    speed=np.array([r['speed_setting_hz'] for r in rows]);load=np.array([r['load_percent'] for r in rows]);y=np.array([r['label'] for r in rows])
    if axis=='speed':
        target=speed==held;source=~target;source_validation=source & (load==75)
    else:
        target=load==held;source=~target;source_validation=source & (speed==45)
    source_training=source & ~source_validation
    keep=~np.array([r.get('excluded_duplicate',False) for r in rows])
    train=source_training & (y<5)&keep;val=source_validation & (y<5)&keep;support=source_training & (y>=5)&keep;target&=keep
    unused=~(train|val|support|target)
    role=np.full(len(rows),'unused_source_novel_validation',dtype=object)
    role[train]='known_train';role[val]='known_val';role[support]='novel_support_pool';role[target]='target_query'
    role[~keep]='excluded_raw_duplicate'
    groups={}
    for row,r in zip(rows,role):
        if r!='excluded_raw_duplicate':groups.setdefault(row['condition_group'],set()).add(r)
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Condition folder leaked between roles')
    if any(np.any(a&b) for i,a in enumerate((train,val,support,target)) for b in (train,val,support,target)[i+1:]):raise ValueError('Role overlap')
    return train,val,support,target,role


def image_scaler(images,train):
    ids=np.flatnonzero(train);s=np.zeros(7);ss=np.zeros(7);count=0
    for begin in range(0,len(ids),64):
        x=np.asarray(images[ids[begin:begin+64]],dtype=np.float64)
        s+=x.sum((0,2,3));ss+=(x*x).sum((0,2,3));count+=len(x)*x.shape[2]*x.shape[3]
    mu=s/count;sd=np.sqrt(np.maximum(ss/count-mu*mu,1e-8))
    return {'mean':mu.tolist(),'std':sd.tolist(),'fit_indices':ids.tolist(),'fit_scope':'source known train only'}


def batch(images,ids,channels,scaler,device):
    raw=images[np.asarray(ids),3:4] if channels==[3] else images[np.asarray(ids)]
    x=np.asarray(raw,dtype=np.float32)
    mu=np.asarray(scaler['mean'],np.float32)[channels];sd=np.asarray(scaler['std'],np.float32)[channels]
    x=(x-mu[None,:,None,None])/sd[None,:,None,None]
    return torch.from_numpy(x).to(device)


class Encoder(nn.Module):
    def __init__(self,channels):
        super().__init__();layers=[]
        for j,(a,b) in enumerate(zip((channels,16,32,64),(16,32,64,128))):
            layers += [nn.Conv2d(a,b,5 if j==0 else 3,stride=2,padding=2 if j==0 else 1,bias=False),
                       nn.GroupNorm(4 if b==16 else 8,b),nn.SiLU()]
        self.body=nn.Sequential(*layers)
        self.project=nn.Sequential(nn.Flatten(),nn.Linear(1024,128),nn.LayerNorm(128))
        self.classifier=nn.Linear(128,5)
    def forward(self,x):
        body=self.body(x)
        # Fixed disjoint bands; avoids nondeterministic adaptive-pool backward.
        pooled=torch.stack([part.mean((2,3)) for part in torch.tensor_split(body,8,dim=2)],dim=2)[:,:,:,None]
        return F.normalize(self.project(pooled),dim=-1)


def balanced_ids(rows,y,train,rng,count_per_class=8):
    # Sample source groups uniformly, then one window; do not overweight a CSV.
    output=[]
    for c in range(5):
        groups={}
        for idx in np.flatnonzero(train & (y==c)):groups.setdefault(rows[idx]['record_id'],[]).append(idx)
        names=sorted(groups)
        chosen=set()
        while len(chosen)<count_per_class:
            candidate=int(rng.choice(groups[names[int(rng.integers(len(names)))]]))
            if candidate not in chosen:chosen.add(candidate);output.append(candidate)
    return np.array(output)


@torch.no_grad()
def encode(model,images,ids,channels,scaler,device,batch_size=64):
    model.eval();out=[]
    for begin in range(0,len(ids),batch_size):out.append(model(batch(images,ids[begin:begin+batch_size],channels,scaler,device)).cpu().numpy())
    return np.concatenate(out)


def train_encoder(rows,images,y,train,val,channels,scaler,args,seed,device,path):
    if (path/'encoder_features.npy').exists() and (path/'best_encoder.pt').exists() and (path/'encoder_history.json').exists():
        history=json.loads((path/'encoder_history.json').read_text(encoding='utf-8'))
        state=torch.load(path/'best_encoder.pt',weights_only=True,map_location='cpu')
        return np.load(path/'encoder_features.npy'),history,sum(t.numel() for t in state.values())
    seed_all(seed);model=Encoder(len(channels)).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.default_rng(seed);history=[];best=float('inf');state=None;vi=np.flatnonzero(val)
    for epoch in range(1,args.encoder_epochs+1):
        model.train();losses=[]
        for _ in range(args.encoder_steps):
            ids=balanced_ids(rows,y,train,rng)
            optimizer.zero_grad(set_to_none=True);z=model(batch(images,ids,channels,scaler,device))
            loss=F.cross_entropy(model.classifier(z),torch.tensor(y[ids],device=device),label_smoothing=args.label_smoothing)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        if epoch%2==0 or epoch==1 or epoch==args.encoder_epochs:
            z=encode(model,images,vi,channels,scaler,device)
            with torch.no_grad():logits=model.classifier(torch.tensor(z,device=device)).cpu().numpy()
            p=softmax(logits,axis=1);score=float(-np.log(p[np.arange(len(vi)),y[vi]].clip(1e-12)).mean())
            history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'source_val_nll':score,'source_val_accuracy':float((p.argmax(1)==y[vi]).mean())})
            if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%10==0:print(f'ENCODER epoch={epoch} source_val_nll={score:.4f}',flush=True)
    path.mkdir(parents=True,exist_ok=True);save_checkpoint(model.state_dict(),path/'last_encoder.pt');save_checkpoint(state,path/'best_encoder.pt')
    dump(path/'encoder_history.json',history);model.load_state_dict(state)
    z=encode(model,images,np.arange(len(rows)),channels,scaler,device)
    np.save(path/'encoder_features.npy',z)
    return z,history,sum(p.numel() for p in model.parameters())


def train_head(features,rows,y,train,val,args,seed,bayesian,device,path):
    if (path/'posterior_features.npy').exists() and (path/'best_head.pt').exists() and (path/'head_history.json').exists():
        history=json.loads((path/'head_history.json').read_text(encoding='utf-8'))
        saved=np.load(path/'posterior_features.npy')
        return torch.tensor(saved,device=device),history,{'parameter_locations':10632,'trainable_parameters':21264 if bayesian else 10632}
    seed_all(seed);mu=features[train].mean(0);sd=features[train].std(0).clip(1e-3)
    x=torch.tensor((features-mu)/sd,dtype=torch.float32,device=device)
    yy=torch.tensor(y,device=device);tr=torch.tensor(train,device=device);vi=np.flatnonzero(val)
    model=FeatureHead(bayesian,args.prior_std).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    rng=np.random.default_rng(seed);history=[];best=float('inf');state=None
    for epoch in range(1,args.head_epochs+1):
        losses=[]
        for _ in range(args.head_steps):
            ids=balanced_ids(rows,y,train,rng,12);query=np.concatenate([np.arange(j*12+5,(j+1)*12) for j in range(5)])
            optimizer.zero_grad(set_to_none=True);total=0.
            for _ in range(args.mc_train if bayesian else 1):
                model.resample();z=model(x[ids]);refs=torch.stack([z[j*12:j*12+5].mean(0) for j in range(5)])
                loss=F.cross_entropy(-torch.cdist(z[query],refs).square()/.07,yy[ids][query],label_smoothing=args.label_smoothing)
                total+=loss
            total/=args.mc_train if bayesian else 1
            total+=args.kl_beta*model.kl()/model.count();total.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(total.detach()))
        with torch.no_grad():
            predictions=[]
            for _ in range(5 if bayesian else 1):
                model.resample();z=model(x);refs=torch.stack([z[tr & (yy==c)].mean(0) for c in range(5)])
                predictions.append((-torch.cdist(z[vi],refs).square()/.07).softmax(-1))
            p=torch.stack(predictions).mean(0);nll=float(-p[torch.arange(len(vi),device=device),yy[vi]].clamp_min(1e-12).log().mean())
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'source_val_nll':nll,
                        'kl_per_parameter':float(model.kl().detach()/model.count())})
        if nll<best:best=nll;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    path.mkdir(parents=True,exist_ok=True);save_checkpoint(model.state_dict(),path/'last_head.pt');save_checkpoint(state,path/'best_head.pt')
    dump(path/'head_history.json',history);dump(path/'feature_scaler.json',{'mean':mu.tolist(),'std':sd.tolist(),
         'fit_indices':np.flatnonzero(train).tolist(),'fit_scope':'source known train only'})
    model.load_state_dict(state);draws=[];weights=[]
    with torch.no_grad():
        for _ in range(args.mc_eval if bayesian else 1):
            model.resample();draws.append(model(x))
            weights.append({f'{i}_{k}':getattr(layer,k+'_sample').detach().cpu() for i,layer in enumerate(model.layers) for k in ('weight','bias')})
    z=torch.stack(draws);save_checkpoint(weights,path/'posterior_weight_samples.pt')
    np.save(path/'posterior_features.npy',z.cpu().numpy())
    return z,history,{'parameter_locations':model.count(),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}


def fixed_distances(z,y,train,mahalanobis=True):
    yy=torch.tensor(y,device=z.device);tr=torch.tensor(train,device=z.device)
    pooled=torch.stack([z[:,tr & (yy==c)].var(1,unbiased=False) for c in range(5)]).mean(0)
    floor=torch.sort(pooled,dim=1).values[:,(pooled.shape[1]-1)//2]*.1
    pooled=torch.maximum(pooled,floor[:,None].clamp_min(1e-5))
    out=[]
    for c in range(5):
        ref=z[:,tr & (yy==c)];delta=z-ref.mean(1)[:,None]
        covariance=.8*pooled+.2*ref.var(1,unbiased=True)
        out.append(torch.sqrt((delta.square()/covariance[:,None].clamp_min(1e-8)).sum(-1)) if mahalanobis else torch.linalg.vector_norm(delta,dim=-1))
    return torch.stack(out,-1),pooled


def append_novel(base,z,pooled,support,mahalanobis=True):
    output=[base]
    for ids in support:
        ref=z[:,ids];delta=z-ref.mean(1)[:,None]
        variance=ref.var(1,unbiased=True) if len(ids)>1 else pooled
        covariance=.8*pooled+.2*variance
        distance=torch.sqrt((delta.square()/covariance[:,None].clamp_min(1e-8)).sum(-1)) if mahalanobis else torch.linalg.vector_norm(delta,dim=-1)
        output.append(distance[:,:,None])
    return torch.cat(output,-1)


def support_ids(rows,y,pool,scenario,seed,k,episode):
    order=SCENARIOS.index(scenario);rng=np.random.default_rng(seed*100000+order*10000+k*100+episode)
    result=[]
    for c in (5,6):
        records={}
        for idx in np.flatnonzero(pool & (y==c)):records.setdefault(rows[idx]['record_id'],[]).append(idx)
        names=sorted(records)
        if k>len(names):raise ValueError('K would exceed distinct available support records')
        chosen=rng.choice(len(names),k,replace=False)
        result.append(np.array([rng.choice(records[names[int(j)]]) for j in chosen],dtype=int))
    return result


def fit_temperature(logits,y):
    def score(logt):
        p=softmax(logits/math.exp(logt),axis=-1).mean(0)
        return float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean())
    opt=minimize_scalar(score,bounds=(math.log(.1),math.log(20.)),method='bounded',options={'maxiter':30})
    return float(math.exp(opt.x))


def metrics(y,p):
    c=p.shape[-1];pred=p.argmax(1);cm=np.bincount(c*y+pred,minlength=c*c).reshape(c,c)
    f1=2*np.diag(cm)/np.maximum(cm.sum(1)+cm.sum(0),1);known=float(f1[:5].mean());novel=float(f1[5:].mean())
    conf=p.max(1);correct=pred==y;ece=0.
    for j in range(10):
        m=(conf>=j/10)&((conf<(j+1)/10) if j<9 else (conf<=1))
        if m.any():ece+=m.mean()*abs(conf[m].mean()-correct[m].mean())
    return {'accuracy':float(correct.mean()),'macro_f1':float(f1.mean()),'base_f1':known,'novel_f1':novel,
        'harmonic_f1':2*known*novel/max(known+novel,1e-12),'per_class_f1':f1.tolist(),'confusion':cm.tolist(),
        'nll':float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean()),
        'brier':float(np.square(p-np.eye(c)[y]).sum(1).mean()),'ece':float(ece),
        'health_accuracy':float(((p[:,0]<.5)==(y!=0)).mean())}


def auc(y,score):
    y=np.asarray(y,bool);r=rankdata(score);n=y.sum();m=(~y).sum()
    return float((r[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def selective_metrics(correct,accepted):
    accepted=np.asarray(accepted,bool);correct=np.asarray(correct,bool)
    return {'coverage':float(accepted.mean()),'selective_error':float(1-correct[accepted].mean()) if accepted.any() else None,
        'accepted_count':int(accepted.sum()),'total_count':len(accepted),
        'correct_confirmed_fraction':float((correct & accepted).mean())}


def threshold_confidence(y,p,target=.05):
    conf=p.max(1);correct=p.argmax(1)==y
    choices=np.unique(np.r_[0.,np.quantile(conf,np.linspace(0,1,41)),1.000001])
    minimum=max(20,int(.1*len(y)))
    for threshold in choices:
        keep=conf>=threshold
        if keep.sum()>=minimum and (1-correct[keep].mean())<=target:return float(threshold)
    return 1.000001


def risk_coverage(y,p):
    correct=p.argmax(1)==y;order=np.argsort(-p.max(1),kind='stable');error=1-correct[order]
    risk=np.cumsum(error)/np.arange(1,len(error)+1)
    grid=np.linspace(.05,1,20);indices=np.maximum((grid*len(error)).astype(int),1)-1
    return {'coverage':grid.tolist(),'risk':risk[indices].tolist(),'aurc':float(risk.mean())}


def sequence_indices(rows,query,max_windows=4):
    groups={}
    for local,idx in enumerate(query):groups.setdefault(rows[idx]['record_id'],[]).append((rows[idx]['window_index'],local))
    result=[]
    for record,pairs in sorted(groups.items()):
        ids=[j for _,j in sorted(pairs)]
        for begin in range(0,len(ids)-max_windows+1,max_windows):result.append((record,np.array(ids[begin:begin+max_windows])))
    return result


def sequential(probabilities,query_distances,y,groups,confidence,radius,mi_limit,bayesian):
    ids=np.stack([indices for _,indices in groups]);n=len(ids);maximum=ids.shape[1]
    previous=np.full(n,-1);decision=np.full(n,-1);lengths=np.full(n,maximum)
    for size in range(1,maximum+1):
        p,h,e,mi=uncertainty(probabilities[:,ids[:,:size]].mean(2))
        pred=p.argmax(1);d=query_distances[ids[:,:size]].mean(1)
        distance=d[np.arange(n),pred]
        take=(decision<0)&(size>=2)&(pred==previous)&(p.max(1)>=confidence)&(distance<=radius)
        if bayesian:take&=mi<=mi_limit
        decision[take]=pred[take];lengths[take]=size;previous=pred
    true=y[ids[:,0]];accepted=decision>=0;correct=decision==true
    return {**selective_metrics(correct,accepted),'mean_windows':float(lengths.mean()),
            'length_histogram':np.bincount(lengths,minlength=maximum+1)[1:maximum+1].tolist(),
            'predictions':decision.tolist(),'true':true.tolist()}


def summarize(evaluations):
    summary={}
    for modality in sorted({r['modality'] for r in evaluations}):
        for method in METHODS:
            for k in sorted({r['k'] for r in evaluations}):
                rows=[r for r in evaluations if r['modality']==modality and r['method']==method and r['k']==k]
                if not rows:continue
                out={};pairs=sorted({(r['scenario'],r['seed']) for r in rows})
                for key in ('macro_f1','base_f1','novel_f1','harmonic_f1','accuracy','ece','nll','brier','health_accuracy'):
                    values=[np.mean([r['calibrated'][key] for r in rows if (r['scenario'],r['seed'])==pair]) for pair in pairs]
                    out[key]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else 0.}
                for key in ('coverage','selective_error','correct_confirmed_fraction'):
                    # Missing accepted error remains null; do not impute zero.
                    values=[r['selective_full'][key] for r in rows if r['selective_full'][key] is not None]
                    out['rejection_'+key]={'mean':float(np.mean(values)) if values else None,'valid_evaluations':len(values)}
                out['units']='support draws averaged within scenario/seed; descriptive, not independent specimens'
                summary[f'{modality}_{method}_k{k}']=out
    return summary


def run(args):
    start=time.perf_counter();torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache=Path(args.cache).resolve()
    if not (cache/'complete.json').exists():prepare(argparse.Namespace(source=DEFAULT_SOURCE,output=cache))
    rows,y,images=load_samples(cache)
    output=Path(args.resume).resolve() if args.resume else Path(args.output).resolve();original=output;i=2
    if not args.resume:
        while output.exists():output=original.with_name(original.name+f'_run{i}');i+=1
        output.mkdir(parents=True)
    elif not (output/'config.json').exists():raise ValueError('Resume directory has no saved config')
    config={**vars(args),'title':TITLE,'output':str(output),'cache':str(cache),'device':str(device),
        'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'torch':torch.__version__,'numpy':np.__version__,
        'classes':CLASSES,'classes_cn':CN,'methods':METHODS,'method_names':NAMES,
        'modalities':list(args.modalities),'modality_channels':MODALITIES,
        'physical_constraints':False,'physical_features':False,'PINN':False,'RUL':False,
        'split_unit':'all four files in each state-condition folder','specimen_id_verified':False,
        'support_domain':'source training conditions','novel_labels_used':'exactly K windows per novel class; two classes, total 2K',
        'target_labels_for_training_calibration_selection':False,'prior':'independent Gaussian N(0,prior_std^2)',
        'posterior':'mean-field Gaussian','kl_normalization':'KL per weight location times kl_beta; tempered VI surrogate',
        'calibration':'known source validation only, temperature range [0.1,20]',
        'rejection':'source empirical risk confidence threshold + source 95th-percentile distance/MI; no target risk guarantee',
        'unknown_task':'K=0 unregistered composite classes, distinct from held-out operating condition',
        'images_saved_automatically':False,'cache_manifest_sha256':sha256(cache/'samples.csv')}
    excluded=[r for r in rows if r['excluded_duplicate']]
    config['duplicate_policy']='Before full experiments, exclude ALL copies of repeated raw SHA256; no labels repaired, no raw files deleted'
    config['excluded_records']=len({r['record_id'] for r in excluded});config['excluded_windows']=len(excluded)
    config['active_records']=len({r['record_id'] for r in rows if not r['excluded_duplicate']})
    config['cuda_deterministic_algorithms']=True
    csv_out(output/'excluded_raw_duplicates.csv',[{key:r[key] for key in ('record_id','class_name','condition_group','raw_sha256')} for r in excluded if r['window_index']==0])
    source_names=('run_laspi_gfsl.py','prepare_laspi.py','bayesian_layers.py')
    config['source_hashes']={name:sha256(ROOT/name) for name in source_names}
    previous_elapsed=0.;elapsed_is_estimate=False
    if args.resume:
        previous=json.loads((output/'config.json').read_text(encoding='utf-8'))
        sensitive=('scenarios','seeds','shots','episodes','encoder_epochs','encoder_steps','head_epochs','head_steps',
                   'mc_train','mc_eval','kl_beta','prior_std','label_smoothing','target_error','cache_manifest_sha256')
        if any(previous[k]!=config[k] for k in sensitive):raise ValueError('Resume scientific configuration changed')
        if sorted(previous['modalities'])!=sorted(args.modalities):raise ValueError('Resume modalities changed')
        config=previous
        def read_saved(name,default):
            path=output/name;return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
        evaluations=read_saved('metrics.json',[]);unknown=read_saved('unknown.json',[]);supports=read_saved('support_ids.json',[])
        model_infos=read_saved('models.json',[]);display=read_saved('display_partial.json',{})
        progress=read_saved('progress.json',{})
        previous_elapsed=progress.get('elapsed_seconds',max(0,(output/'metrics.json').stat().st_mtime-(output/'config.json').stat().st_mtime))
        elapsed_is_estimate=not bool(progress)
        events=read_saved('resume_history.json',[]);snap=output/f'source_snapshot_resume_{len(events)+1}';snap.mkdir()
        for name in source_names:shutil.copy2(ROOT/name,snap/name)
        events.append({'source_hashes':{name:sha256(ROOT/name) for name in source_names},'snapshot':str(snap),
             'completed_head_models':len(model_infos),'existing_evaluations':len(evaluations),
             'changes':'Checkpoint file-stream atomic save and resume; scientific parameters unchanged',
             'previous_elapsed_estimated':elapsed_is_estimate})
        dump(output/'resume_history.json',events)
    else:
        snap=output/'source_snapshot';snap.mkdir()
        for name in source_names:shutil.copy2(ROOT/name,snap/name)
        dump(output/'config.json',config);evaluations=[];unknown=[];supports=[];model_infos=[];display={}
    for scenario in args.scenarios:
        train,val,pool,target,role=masks(rows,scenario);vi=np.flatnonzero(val);ti=np.flatnonzero(target)
        csv_out(output/f'splits_{scenario}.csv',[{**r,'role':str(role[i])} for i,r in enumerate(rows)])
        scaler_file=output/f'image_scaler_{scenario}.json'
        if args.resume and scaler_file.exists():
            scaler=json.loads(scaler_file.read_text(encoding='utf-8'))
            if scaler['fit_indices']!=np.flatnonzero(train).tolist():raise ValueError('Resume scaler fit scope changed')
        else:scaler=image_scaler(images,train);dump(scaler_file,scaler)
        sequence_groups=sequence_indices(rows,ti)
        for seed in args.seeds:
            for k in args.shots:
                for episode in range(args.episodes):
                    ids=support_ids(rows,y,pool,scenario,seed,k,episode)
                    item={'scenario':scenario,'seed':seed,'k':k,'episode':episode,'per_class_ids':[[rows[i]['sample_id'] for i in a] for a in ids]}
                    found=[s for s in supports if (s['scenario'],s['seed'],s['k'],s['episode'])==(scenario,seed,k,episode)]
                    if found:
                        if found[0]!=item:raise ValueError('Resume support budget changed')
                    else:supports.append(item)
            for modality in args.modalities:
                channels=MODALITIES[modality];destination=output/'models'/f'{scenario}_{modality}_s{seed}'
                completed=[r for r in evaluations if r['scenario']==scenario and r['seed']==seed and r['modality']==modality]
                expected=len(args.shots)*args.episodes*3
                if len(completed)==expected and all((output/'predictions'/f'{scenario}_{modality}_{m}_s{seed}.npz').exists() for m in METHODS):
                    print(f'REUSED_COMPLETE {scenario} {modality} seed={seed}',flush=True);continue
                features,encoder_history,encoder_count=train_encoder(rows,images,y,train,val,channels,scaler,args,seed,device,destination)
                print(f'ENCODER_DONE {scenario} {modality} seed={seed}',flush=True)
                for bayesian,head_name in ((False,'deterministic'),(True,'bayesian')):
                    z,history,count=train_head(features,rows,y,train,val,args,seed,bayesian,device,destination/head_name)
                    model_infos=[r for r in model_infos if (r['scenario'],r['seed'],r['modality'],r['head'])!=(scenario,seed,modality,head_name)]
                    model_infos.append({'scenario':scenario,'seed':seed,'modality':modality,'head':head_name,
                        'encoder_parameters':encoder_count,**count,'best_encoder_epoch':min(encoder_history,key=lambda a:a['source_val_nll'])['epoch'],
                        'best_head_epoch':min(history,key=lambda a:a['source_val_nll'])['epoch']})
                    methods=['bayesian_mahalanobis'] if bayesian else ['euclidean','mahalanobis']
                    for method in methods:
                        evaluations=[r for r in evaluations if (r['scenario'],r['seed'],r['modality'],r['method'])!=(scenario,seed,modality,method)]
                        unknown=[r for r in unknown if (r['scenario'],r['seed'],r['modality'],r['method'])!=(scenario,seed,modality,method)]
                        maha=method!='euclidean';base,pooled=fixed_distances(z,y,train,maha)
                        # K=0: novel composites were not registered; calibrate five candidates.
                        vl=-base[:,vi].cpu().numpy();tt=fit_temperature(vl,y[vi]);pv=softmax(vl/tt,axis=-1)
                        ql=-base[:,ti].cpu().numpy();pq=softmax(ql/tt,axis=-1)
                        vmean,_,_,vmi=uncertainty(pv);qmean,_,_,qmi=uncertainty(pq)
                        vr=base[:,vi].mean(0).min(-1).values.cpu().numpy();qr=base[:,ti].mean(0).min(-1).values.cpu().numpy()
                        radius0=max(float(np.quantile(vr,.95)),1e-6);mi0=max(float(np.quantile(vmi,.95)),1e-6)
                        score=qr/radius0;combined=np.maximum(score,qmi/mi0) if bayesian else score
                        is_novel=y[ti]>=5
                        unknown.append({'scenario':scenario,'seed':seed,'modality':modality,'method':method,'k':0,
                            'distance_auroc':auc(is_novel,score),'distance_mi_auroc':auc(is_novel,combined),
                            'unknown_false_acceptance':float((combined[is_novel]<=1).mean()),
                            'known_coverage':float((combined[~is_novel]<=1).mean()),'radius':radius0,'mi_limit':mi0})
                        blocks=[];uq_blocks=[];temps=[];selection_blocks=[];support_indices=[]
                        for k in args.shots:
                            for episode in range(args.episodes):
                                ids=support_ids(rows,y,pool,scenario,seed,k,episode);dist=append_novel(base,z,pooled,ids,maha)
                                v_logits=-dist[:,vi].cpu().numpy();q_logits=-dist[:,ti].cpu().numpy()
                                temperature=fit_temperature(v_logits,y[vi])
                                vprob=softmax(v_logits/temperature,axis=-1);qprob=softmax(q_logits/temperature,axis=-1)
                                p,entropy,expected,mi=uncertainty(qprob);v,_,_,vmi=uncertainty(vprob)
                                raw=softmax(q_logits,axis=-1).mean(0)
                                conf=threshold_confidence(y[vi],v,args.target_error)
                                val_distance=dist[:,vi].mean(0).cpu().numpy();query_distance=dist[:,ti].mean(0).cpu().numpy()
                                radius=float(np.quantile(val_distance[np.arange(len(vi)),y[vi]],.95));limit=float(np.quantile(vmi,.95))
                                prediction=p.argmax(1);conf_accept=p.max(1)>=conf
                                dist_accept=query_distance[np.arange(len(ti)),prediction]<=radius
                                full_accept=conf_accept & dist_accept & ((mi<=limit) if bayesian else True)
                                record={'scenario':scenario,'seed':seed,'modality':modality,'method':method,'k':k,'episode':episode,
                                    'temperature':temperature,'confidence_threshold':conf,'radius':radius,'mi_limit':limit,
                                    'raw':metrics(y[ti],raw),'calibrated':metrics(y[ti],p),
                                    'selective_confidence':selective_metrics(prediction==y[ti],conf_accept),
                                    'selective_full':selective_metrics(prediction==y[ti],full_accept),
                                    'sequential':sequential(qprob,query_distance,y[ti],sequence_groups,conf,radius,limit,bayesian),
                                    'risk_coverage':risk_coverage(y[ti],p),'mean_mi':float(mi.mean())}
                                evaluations.append(record);blocks.append(p.astype(np.float64))
                                uq_blocks.append(np.column_stack([entropy,expected,mi]).astype(np.float32));temps.append(temperature)
                                selection_blocks.append(np.column_stack([conf_accept,full_accept]));support_indices.append(np.r_[ids[0],ids[1]])
                                if scenario==args.scenarios[0] and seed==args.seeds[0] and k==5 and episode==0:
                                    display[f'{modality}_{method}']={**record,'encoder_history':encoder_history,'head_history':history,
                                        'query_indices':ti.tolist(),'probabilities':p.tolist(),'entropy':entropy.tolist(),'mi':mi.tolist(),
                                        'accepted':full_accept.tolist()}
                        path=output/'predictions';path.mkdir(exist_ok=True)
                        save_predictions(path/f'{scenario}_{modality}_{method}_s{seed}.npz',
                            probabilities=np.stack(blocks),uncertainties=np.stack(uq_blocks),query_indices=ti,
                            temperatures=np.array(temps),acceptance=np.stack(selection_blocks),true_labels=y[ti],
                            shots=np.repeat(args.shots,args.episodes),episodes=np.tile(np.arange(args.episodes),len(args.shots)))
                        del base,pooled
                    print(f'EVALUATED {scenario} {modality} {head_name} seed={seed}',flush=True)
                    dump(output/'metrics.json',evaluations);dump(output/'support_ids.json',supports);dump(output/'unknown.json',unknown)
                    dump(output/'models.json',model_infos);del z
                    dump(output/'display_partial.json',display)
                    dump(output/'progress.json',{'elapsed_seconds':previous_elapsed+time.perf_counter()-start,
                        'previous_elapsed_estimated':elapsed_is_estimate,'head_models':len(model_infos),'evaluations':len(evaluations)})
                    if device.type=='cuda':torch.cuda.empty_cache()
    dump(output/'summary.json',summarize(evaluations));dump(output/'display.json',display)
    dump(output/'execution.json',{'elapsed_seconds':previous_elapsed+time.perf_counter()-start,
        'previous_elapsed_estimated':elapsed_is_estimate,'resume_seconds':time.perf_counter()-start,'encoder_models':len(model_infos)//2,
        'head_models':len(model_infos),'registered_evaluations':len(evaluations),'actual_support_sets':len(supports),
        'K0_evaluations':len(unknown),'completed':True})
    print('FINISHED_LASPI',output,flush=True)
    return output


def arguments():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,default=ROOT/'prepared');ap.add_argument('--output',type=Path,default=ROOT/'results'/'laspi_cross_condition_20261012')
    ap.add_argument('--resume',type=Path,help='Continue an interrupted run with identical scientific configuration')
    ap.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS));ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    ap.add_argument('--modalities',nargs='+',choices=list(MODALITIES),default=list(MODALITIES))
    ap.add_argument('--shots',type=int,nargs='+',default=[1,5,10]);ap.add_argument('--episodes',type=int,default=20)
    ap.add_argument('--encoder-epochs',type=int,default=25);ap.add_argument('--encoder-steps',type=int,default=20)
    ap.add_argument('--head-epochs',type=int,default=25);ap.add_argument('--head-steps',type=int,default=10)
    ap.add_argument('--mc-train',type=int,default=2);ap.add_argument('--mc-eval',type=int,default=20)
    ap.add_argument('--kl-beta',type=float,default=.01);ap.add_argument('--prior-std',type=float,default=1.)
    ap.add_argument('--label-smoothing',type=float,default=.05);ap.add_argument('--target-error',type=float,default=.05)
    ap.add_argument('--no-show',action='store_true');ap.add_argument('--skip-condition-correction',action='store_true')
    args=ap.parse_args()
    if min(args.episodes,args.encoder_epochs,args.encoder_steps,args.head_epochs,args.head_steps,args.mc_train,args.mc_eval)<=0:raise ValueError('Budgets must be positive')
    if args.prior_std<=0 or args.kl_beta<0 or not 0<=args.label_smoothing<1:raise ValueError('Invalid prior/KL/smoothing')
    return args


if __name__=='__main__':
    args=arguments();result=run(args)
    correction=None
    if not args.skip_condition_correction:
        from evaluate_condition_correction import evaluate
        candidate=result/'condition_correction'
        if (candidate/'execution.json').exists():correction=candidate
        else:correction=evaluate(result)
    from analyze_laspi import analyze
    analyze(result,correction)
    if not args.no_show:
        from show_laspi_results import show_results
        show_results(result)
