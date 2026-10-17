"""Run repaired public-code DGGN adaptation on CSV-disjoint LASPI few-shot tasks."""
import argparse,copy,csv,json,subprocess,sys,time,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.special import softmax
from prepare_laspi import ROOT,dump,csv_out,sha256,normalize_windows
from run_laspi_gfsl import load_samples,seed_all,save_checkpoint,save_predictions,metrics
from run_laspi_cross_702010 import split_cross,cross_supports
from temporal_information import reserve_calibration
from mahalanobis_metric import fit_reference,mahalanobis_logits
from dggn_laspi_adapter import AdaptedDGGN,FixedLogFFT,train_loss,embeddings,boundary_exemplars

def prepare_waveforms(cfg,rows):
    folder=ROOT/cfg.waveform_cache;marker=folder/'complete.json'
    expected={'fs':25600,'length':8192,'windows_per_csv':31,'normalization':'window_zscore','reference_cache':cfg.reference_cache}
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8'))['contract']!=expected:raise ValueError('波形缓存配置不匹配')
        with (folder/'waveforms.npy').open('rb') as stream:return np.load(stream,allow_pickle=False)
    if folder.exists():raise FileExistsError('未完成波形缓存保留，请检查或改waveform_cache：'+str(folder))
    folder.mkdir(parents=True);wave=np.lib.format.open_memmap(folder/'waveforms.npy',mode='w+',dtype=np.float16,shape=(len(rows),8192));audits=[]
    with (ROOT/cfg.reference_cache/'records.csv').open(encoding='utf-8-sig') as stream:records=list(csv.DictReader(stream))
    by_record={}
    for i,r in enumerate(rows):by_record.setdefault(r['record_id'],[]).append(i)
    for record in tqdm(records,desc='读取原采样率振动',unit='CSV',dynamic_ncols=True):
        if sha256(record['file'])!=record['sha256']:raise ValueError('原始文件SHA变化：'+record['file'])
        signal=np.loadtxt(record['file'],delimiter=',',usecols=[3],dtype=np.float32)
        if signal.shape!=(256000,) or not np.isfinite(signal).all():raise ValueError('波形形状/数值错误')
        chunks=normalize_windows(signal[:31*8192].reshape(31,8192),'window_zscore')
        ids=sorted(by_record[record['record_id']],key=lambda i:rows[i]['window_index']);wave[ids]=chunks.astype(np.float16)
        audits.append({'record_id':record['record_id'],'sha256':record['sha256'],'windows':31})
    wave.flush();del wave;dump(marker,{'contract':expected,'records':audits,'source':'raw CSV vibration column 3; no downsampling; 0.32 seconds/shot'})
    with (folder/'waveforms.npy').open('rb') as stream:return np.load(stream,allow_pickle=False)

def batch(wave,ids,device):return torch.from_numpy(wave[np.asarray(ids)].astype(np.float32)).to(device)

def balanced_ids(pool,y,rng,count=8):
    return np.concatenate([rng.choice(pool[y[pool]==c],count,replace=len(pool[y[pool]==c])<count) for c in np.unique(y[pool])])

@torch.no_grad()
def source_score(model,wave,ids,y,device,batch_size):
    model.eval();loss=[];prob=[]
    for start in range(0,len(ids),batch_size):
        chosen=ids[start:start+batch_size];_,_,logits,_=model.outputs(batch(wave,chosen,device));loss.append(float(F.cross_entropy(logits,torch.tensor(y[chosen],device=device)))*len(chosen));prob.append(logits.softmax(-1).cpu().numpy())
    p=np.concatenate(prob);return {'nll':sum(loss)/len(ids),'accuracy':float((p.argmax(1)==y[ids]).mean())}

def fit_source(wave,rows,y,role,cfg,seed,folder,device):
    seed_all(seed);model=AdaptedDGGN(ROOT/cfg.repository,wave.shape[1],device,getattr(cfg,'input_encoding','raw'))
    train=np.flatnonzero(role=='base_train');val=np.flatnonzero(role=='base_val');history=[];best=float('inf')
    if getattr(cfg,'input_encoding','raw')=='logfft':
        total=np.zeros(1024);squared=np.zeros(1024);count=0
        with torch.no_grad():
            for start in range(0,len(train),cfg.batch_size):
                z=FixedLogFFT.transform(batch(wave,train[start:start+cfg.batch_size],device)).cpu().numpy().astype(np.float64)
                total+=z.sum(0);squared+=(z*z).sum(0);count+=len(z)
        mean=total/count;std=np.sqrt(np.maximum(squared/count-mean*mean,1e-6))
        for net in (model.ca,model.cs):
            net.frontend.mean.copy_(torch.tensor(mean,dtype=torch.float32,device=device));net.frontend.std.copy_(torch.tensor(std,dtype=torch.float32,device=device))
        dump(folder/'spectral_scaler.json',{'mean':mean.tolist(),'std':std.tolist(),'fit_indices':train.tolist(),'fit_scope':'source base_train only'})
    rng=np.random.default_rng(seed);generator=torch.Generator(device=device);generator.manual_seed(seed+21)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.source_lr,weight_decay=cfg.weight_decay)
    for epoch in tqdm(range(1,cfg.source_epochs+1),desc='DGGN源训练 '+folder.name,unit='轮',dynamic_ncols=True):
        losses=[]
        for _ in range(cfg.steps_per_epoch):
            model.train();ids=balanced_ids(train,y,rng,cfg.source_per_class);optimizer.zero_grad(set_to_none=True)
            loss,info=train_loss(model,batch(wave,ids,device),torch.tensor(y[ids],device=device),cfg,generator)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        score=source_score(model,wave,val,y,device,cfg.batch_size);history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),**score})
        if score['nll']<best:best=score['nll'];save_checkpoint(model.state_dict(),folder/'source.pt')
        dump(folder/'source_history.json',history)
        if epoch==1 or epoch%10==0:print(f'源训练 {epoch}/{cfg.source_epochs}: loss={np.mean(losses):.4f}, 源验证Acc={score["accuracy"]:.2%}, NLL={score["nll"]:.4f}',flush=True)
    model.load_state_dict(torch.load(folder/'source.pt',weights_only=True,map_location=device));model.eval();return model

def classifier_scores(train_features,train_labels,query_features,query_labels,cfg,seed,device):
    forest=BalancedRandomForestClassifier(n_estimators=cfg.rf_trees,random_state=seed,sampling_strategy='auto',replacement=True,bootstrap=True,n_jobs=2)
    forest.fit(train_features,train_labels);prob=forest.predict_proba(query_features)
    padded=np.zeros((len(query_features),7));padded[:,forest.classes_.astype(int)]=prob
    tr=torch.tensor(train_features,device=device);labels=torch.tensor(train_labels,device=device);centers,chol=fit_reference(tr,labels,classes=7,shrinkage=cfg.shrinkage)
    maha=mahalanobis_logits(torch.tensor(query_features,device=device),centers,chol).detach().cpu().numpy();mp=softmax(maha,axis=-1)
    return {'brf':metrics(query_labels,padded),'mahalanobis':metrics(query_labels,mp)},padded,mp

def train_task(source,wave,y,role,support,memory,cfg,task_seed,device):
    seed_all(task_seed);student=source.incremental_copy();teacher=copy.deepcopy(source).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=cfg.adapt_lr,weight_decay=cfg.weight_decay)
    rng=np.random.default_rng(task_seed);generator=torch.Generator(device=device);generator.manual_seed(task_seed+33)
    novel=np.array(support['indices'][0]+support['indices'][1]);history=[];frozen_before={k:v.clone() for k,v in student.old_cs.state_dict().items()}
    for step in range(cfg.adapt_steps):
        replay=balanced_ids(memory,y,rng,cfg.replay_per_class);ids=np.r_[replay,novel];student.train();optimizer.zero_grad(set_to_none=True)
        loss,info=train_loss(student,batch(wave,ids,device),torch.tensor(y[ids],device=device),cfg,generator,teacher)
        loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),5);optimizer.step();history.append({'step':step+1,'loss':float(loss.detach()),**info})
    if any(not torch.equal(v,frozen_before[k]) for k,v in student.old_cs.state_dict().items()):raise AssertionError('旧类别分支或BN状态被修改')
    student.eval();return student,history

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'dggn_laspi_pilot.json');parser.add_argument('--check',action='store_true');parser.add_argument('--resume',type=Path);args=parser.parse_args()
    settings=json.loads(args.config.read_text(encoding='utf-8'));cfg=SimpleNamespace(**settings)
    if max(cfg.shots)>10 or min(cfg.shots)<1:raise ValueError('仅1/5/10-shot，最多10')
    if cfg.memory_per_base_class<cfg.replay_per_class or min(cfg.source_epochs,cfg.adapt_steps,cfg.steps_per_epoch)<1:raise ValueError('内存/训练预算错误')
    for p in ('model.py','nets/feature_net_mff.py'):
        if not (ROOT/cfg.repository/p).exists():raise FileNotFoundError(p)
    rows,y,_=load_samples(ROOT/cfg.reference_cache,metadata_only=True)
    for case in cfg.cases:
        for seed in cfg.seeds:
            role,*_=split_cross(rows,case,seed);role,reserved=reserve_calibration(rows,y,role,seed)
            for k in cfg.shots:cross_supports(rows,y,role,case,seed,k,1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('设备='+str(device)+'；设置='+str(settings),flush=True)
    if args.check:return
    torch.set_num_threads(2);started=time.perf_counter()
    if args.resume:
        root=args.resume.resolve()
        if json.loads((root/'config.json').read_text(encoding='utf-8'))!=settings:raise ValueError('续跑配置与原实验不一致')
    else:
        root=ROOT/cfg.output;original=root;number=2
        while root.exists():root=original.with_name(original.name+f'_run{number}');number+=1
        root.mkdir(parents=True);dump(root/'config.json',settings)
        snapshot=root/'source_snapshot';snapshot.mkdir()
        for name in ('dggn_laspi_adapter.py','run_dggn_laspi.py','show_dggn_laspi.py'):shutil.copy2(ROOT/name,snapshot/name)
        dump(root/'provenance.json',{'repository':str(ROOT/cfg.repository),'upstream_files':{name:sha256(ROOT/cfg.repository/name) for name in ('model.py','nets/feature_net_mff.py')},'status':'repaired public-code adaptation, not exact paper replication','input_encoding':getattr(cfg,'input_encoding','raw'),'input_change':'23 original features -> 8192 normalized raw samples, optionally fixed 1024-bin logFFT with source-only scaler; public dense/MOIA backbone retained','augmentation_change':'feature reorder -> circular waveform shift and small additive noise, applied in raw waveform domain for BOTH input encodings','classifier':'public-code CA embeddings -> balanced random forest; same-memory Mahalanobis reported separately','bn_policy':'training support/replay only; frozen teacher and old-CS BatchNorm fixed','ca_replay_distillation':'teacher encoder features, not class logits','classes':'base 0..4, add both composites 5,6 in one incremental session'})
    dump(root/'execution.json',{'finished':False,'device':str(device)});print('RESULT_DIR='+str(root),flush=True)
    wave=prepare_waveforms(cfg,rows);records=[]
    for case in cfg.cases:
        for seed in cfg.seeds:
            role,_,_,_,audit=split_cross(rows,case,seed);role,reserved=reserve_calibration(rows,y,role,seed);folder=root/f'{case}_s{seed}';folder.mkdir(exist_ok=True)
            csv_out(folder/'split.csv',[{'sample_id':r['sample_id'],'record_id':r['record_id'],'role':role[i],'label':int(y[i])} for i,r in enumerate(rows)]);dump(folder/'split_audit.json',{**audit,'reserved_validation_csvs':reserved,'target_fit':False})
            if (folder/'SOURCE_FINISHED.json').exists():
                source=AdaptedDGGN(ROOT/cfg.repository,wave.shape[1],device,getattr(cfg,'input_encoding','raw'));source.load_state_dict(torch.load(folder/'source.pt',map_location=device,weights_only=True));source.eval()
            else:
                source=fit_source(wave,rows,y,role,cfg,seed,folder,device);dump(folder/'SOURCE_FINISHED.json',{'finished':True,'epochs':cfg.source_epochs})
            tr=np.flatnonzero(role=='base_train');query=np.flatnonzero(role=='query');base_features=embeddings(source,wave,tr,device,cfg.batch_size)
            memory=np.concatenate([boundary_exemplars(base_features[y[tr]==c],tr[y[tr]==c],cfg.memory_per_base_class) for c in range(5)])
            dump(folder/'memory.json',{'indices':memory.tolist(),'sample_ids':[rows[i]['sample_id'] for i in memory],'strategy':'BAEP far-residual ordering, row-normalized embeddings, aligned dataset indices','per_base_class':cfg.memory_per_base_class})
            frozen_source={k:v.clone() for k,v in source.state_dict().items()};tasks=folder/'tasks';tasks.mkdir(exist_ok=True)
            for k in cfg.shots:
                for support in tqdm(cross_supports(rows,y,role,case,seed,k,cfg.support_repeats),desc=f'DGGN {case} s{seed} {k}-shot',unit='任务',dynamic_ncols=True):
                    episode=support['episode'];path=tasks/f'k{k}_e{episode}.json'
                    if path.exists():records.append(json.loads(path.read_text(encoding='utf-8')));continue
                    task_seed=seed*100000+k*100+episode;student,history=train_task(source,wave,y,role,support,memory,cfg,task_seed,device)
                    if any(not torch.equal(v,frozen_source[n]) for n,v in source.state_dict().items()):raise AssertionError('源模型被适配修改')
                    reference=np.r_[memory,support['indices'][0],support['indices'][1]];features=embeddings(student,wave,np.r_[reference,query],device,cfg.batch_size)
                    scores,rf,maha=classifier_scores(features[:len(reference)],y[reference],features[len(reference):],y[query],cfg,task_seed,device)
                    record={'finished':True,'condition':case,'seed':seed,'k':k,'episode':episode,'metrics':scores,'support_sample_ids':support['sample_ids'],'source_unchanged':True,'old_cs_unchanged':True,'train_loss_first':history[0]['loss'],'train_loss_last':history[-1]['loss'],'classifier_reference_counts':{str(c):int((y[reference]==c).sum()) for c in range(7)}}
                    if episode==0:
                        save_checkpoint(student.state_dict(),folder/f'adapted_{k}shot.pt');dump(folder/f'adapt_history_{k}shot.json',history)
                        save_predictions(folder/f'predictions_{k}shot.npz',brf_probabilities=rf,mahalanobis_probabilities=maha,true_labels=y[query],query_indices=query)
                    dump(path,record);records.append(record)
                    print(f'{case} {k}-shot #{episode+1}: BRF MacroF1={scores["brf"]["macro_f1"]:.3f}, 新类F1={scores["brf"]["novel_f1"]:.3f}；马氏 MacroF1={scores["mahalanobis"]["macro_f1"]:.3f}',flush=True)
                    del student
            del source
    dump(root/'metrics.json',records);table=[]
    for case in cfg.cases:
        for k in cfg.shots:
            group=[r for r in records if r['condition']==case and r['k']==k]
            for classifier in ('brf','mahalanobis'):
                table.append({'condition':case,'k':k,'classifier':classifier,'tasks':len(group),**{key:float(np.mean([r['metrics'][classifier][key] for r in group])) for key in ('accuracy','macro_f1','base_f1','novel_f1','ece')}})
    dump(root/'summary.json',table);csv_out(root/'summary.csv',table)
    dump(root/'execution.json',{'finished':True,'device':str(device),'evaluations':len(records),'elapsed_seconds':time.perf_counter()-started})
    print('DGGN适配实验完成：'+str(root),flush=True)
    if cfg.show_plots:
        from show_dggn_laspi import show
        show(root)

if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    main()
