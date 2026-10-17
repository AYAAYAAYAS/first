"""Four LWC ablations with source-only stability and paired K-shot adaptation."""
import argparse,copy,json,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax,expit
from tqdm.auto import tqdm
from prepare_laspi import ROOT,dump,csv_out
from run_laspi_gfsl import load_samples,seed_all,save_checkpoint,save_predictions,fit_temperature,metrics
from run_laspi_cross_702010 import split_cross,cross_supports
from temporal_information import reserve_calibration
from train_laspi_components import SIGNATURES
from run_dggn_laspi import prepare_waveforms,batch
from mahalanobis_metric import fit_reference,mahalanobis_logits
from lwc_stability_model import MODES,DualLWC,configure_training,source_objective,support_objective

def sample_cross_speed(pool,rows,y,rng,count=4):
    ids=[];support=[];query=[];audit=[]
    speed=np.array([r['speed_setting_hz'] for r in rows]);load=np.array([r['load_percent'] for r in rows])
    for c in range(5):
        candidates=[]
        for l in np.unique(load[pool]):
            domains=[d for d in np.unique(speed[pool]) if ((y[pool]==c)&(load[pool]==l)&(speed[pool]==d)).any()]
            if len(domains)>=2:candidates.append((l,domains))
        if not candidates:raise ValueError('源样本无法形成同故障同载荷跨转速配对')
        l,domains=candidates[int(rng.integers(len(candidates)))];a,b=rng.choice(domains,2,replace=False)
        ia=pool[(y[pool]==c)&(load[pool]==l)&(speed[pool]==a)];ib=pool[(y[pool]==c)&(load[pool]==l)&(speed[pool]==b)]
        start=len(ids);ids.extend(rng.choice(ia,count,replace=len(ia)<count));ids.extend(rng.choice(ib,count,replace=len(ib)<count))
        support.extend(range(start,start+count));query.extend(range(start+count,start+2*count));audit.append({'class':c,'load':int(l),'speed_groups':[int(a),int(b)]})
    return np.array(ids),np.array(support),np.array(query),audit

def fixed_memory(rows,y,role,seed,count):
    rng=np.random.default_rng(seed+519);ids=[]
    for c in range(5):
        groups={}
        for i in np.flatnonzero((role=='base_train')&(y==c)):groups.setdefault(rows[i]['record_id'],[]).append(i)
        names=sorted(groups)
        if count>len(names):raise ValueError('每基类独立CSV数量不足以建立参考记忆')
        ids.extend(int(rng.choice(groups[names[j]])) for j in rng.choice(len(names),count,replace=False))
    return np.array(ids)

def record_prefix(rows,role,name,count):
    return np.array([i for i,r in enumerate(rows) if role[i]==name and r['window_index']<count])

@torch.no_grad()
def extract(model,wave,ids,device,batch_size):
    model.eval();pieces={};gates=[]
    for start in range(0,len(ids),batch_size):
        out=model(batch(wave,ids[start:start+batch_size],device))
        for key in ('d','s','d_logits','s_logits'):
            if out[key] is not None:pieces.setdefault(key,[]).append(out[key].cpu().numpy())
        gates.append(float(out['gate']))
    return {**{k:np.concatenate(v) for k,v in pieces.items()},'gate':float(np.mean(gates))}

def classify(values,reference_local,query_local,labels,cfg,novel_local=None):
    scores=[]
    with torch.no_grad():
        for key in ('d','s'):
            if key not in values:continue
            x=torch.tensor(values[key]);means,chol=fit_reference(x[reference_local],torch.tensor(labels),classes=5,shrinkage=cfg.shrinkage)
            if novel_local is not None:means=torch.cat([means,torch.stack([x[v].mean(0) for v in novel_local])])
            scores.append(mahalanobis_logits(x[query_local],means,chol).numpy()/cfg.metric_temperature)
    return np.mean(scores,axis=0)

def source_validation(model,wave,memory,val,y,cfg,device):
    values=extract(model,wave,np.r_[memory,val],device,cfg.batch_size)
    logits=classify(values,np.arange(len(memory)),np.arange(len(memory),len(memory)+len(val)),y[memory],cfg)
    p=softmax(logits,axis=-1);return {'nll':float(-np.log(p[np.arange(len(val)),y[val]].clip(1e-12)).mean()),'accuracy':float((p.argmax(1)==y[val]).mean()),'gate':values['gate']}

def train_source(model,wave,rows,y,role,memory,cfg,seed,folder,phase):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'finished.json').exists():
        model.load_state_dict(torch.load(folder/'best.pt',weights_only=True,map_location=cfg.device));return model
    optimizer=configure_training(model,cfg,'pretrain' if phase=='pretrain' else 'source')
    tr=np.flatnonzero(role=='base_train');val=record_prefix(rows,role,'base_val',cfg.validation_windows_per_csv)
    rng=np.random.default_rng(seed+107);speeds=torch.tensor([r['speed_setting_hz'] for r in rows],device=cfg.device)
    epochs=cfg.source_epochs if phase=='pretrain' else cfg.exchange_epochs;history=[];best=float('inf')
    for epoch in tqdm(range(1,epochs+1),desc=f'LWC {phase} {model.mode}',unit='轮',dynamic_ncols=True,mininterval=1):
        losses=[];align=[]
        for step in range(cfg.steps_per_epoch):
            ids,s,q,audit=sample_cross_speed(tr,rows,y,rng,cfg.source_per_domain)
            model.train();optimizer.zero_grad(set_to_none=True)
            loss,info=source_objective(model,batch(wave,ids,cfg.device),torch.tensor(y[ids],device=cfg.device),torch.tensor(SIGNATURES[y[ids]],device=cfg.device),speeds[ids],s,q,cfg,pretrain=phase=='pretrain')
            if not torch.isfinite(loss):raise FloatingPointError('源训练损失非有限')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()));align.append(info['alignment'])
            if epoch==1 and step==0:dump(folder/'first_speed_pair_audit.json',{'pairs':audit,'sample_ids':[rows[i]['sample_id'] for i in ids]})
        score=source_validation(model,wave,memory,val,y,cfg,cfg.device);history.append({'epoch':epoch,'loss':float(np.mean(losses)),'alignment':float(np.mean(align)),**score})
        if score['nll']<best:best=score['nll'];save_checkpoint(model.state_dict(),folder/'best.pt')
        dump(folder/'history.json',history)
        if epoch==1 or epoch%10==0:print(f'{phase}/{model.mode} {epoch}/{epochs}: loss={np.mean(losses):.4f}, 验证Acc={score["accuracy"]:.2%}, 稳定中心差={np.mean(align):.4f}, gate={score["gate"]:.4f}',flush=True)
    model.load_state_dict(torch.load(folder/'best.pt',weights_only=True,map_location=cfg.device));dump(folder/'finished.json',{'finished':True,'epochs':epochs,'fit_scope':'source base only'});return model

def state_change(before,module):return max(float((v.detach().cpu()-before[k]).abs().max()) for k,v in module.state_dict().items())

def adapt(source,wave,rows,y,memory,support,cfg,task_seed,mode):
    seed_all(task_seed);model=copy.deepcopy(source);model.mode=mode;teacher=copy.deepcopy(source).eval();teacher.mode=mode
    for p in teacher.parameters():p.requires_grad_(False)
    stable_before={k:v.detach().cpu().clone() for k,v in model.stable.state_dict().items()}
    front_before={k:v.detach().cpu().clone() for k,v in model.discriminative.wavelet.state_dict().items()}
    optimizer=configure_training(model,cfg,'support');rng=np.random.default_rng(task_seed+201)
    generator=torch.Generator(device=cfg.device);generator.manual_seed(task_seed+415)
    novel=np.array(support['indices'][0]+support['indices'][1]);history=[]
    for step in range(cfg.adapt_steps):
        replay,_,_,_=sample_cross_speed(memory,rows,y,rng,cfg.replay_per_domain)
        ids=np.r_[replay,novel];x=batch(wave,ids,cfg.device);labels=torch.tensor(y[ids],device=cfg.device);targets=torch.tensor(SIGNATURES[y[ids]],device=cfg.device);speeds=torch.tensor([rows[i]['speed_setting_hz'] for i in ids],device=cfg.device)
        noise=cfg.reference_noise*torch.randn(x.shape,generator=generator,device=cfg.device)
        model.train();optimizer.zero_grad(set_to_none=True)
        loss,info=support_objective(model,teacher,x,labels,targets,speeds,cfg,noise)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();history.append({'step':step+1,'loss':float(loss.detach()),**info})
    stable_delta=state_change(stable_before,model.stable);front_delta=state_change(front_before,model.discriminative.wavelet)
    if mode!='trainable_exchange' and stable_delta!=0:raise AssertionError('冻结稳定分支发生参数漂移')
    if mode=='trainable_exchange' and stable_delta==0:raise AssertionError('可训练稳定分支没有更新')
    if front_delta!=0:raise AssertionError('判别LWC前端在少样本阶段被意外更新')
    return model,history,{'stable_max_change':stable_delta,'discriminative_wavelet_max_change':front_delta}

def stability_diagnostic(z,labels,speeds):
    within=[];centers=[]
    for c in np.unique(labels):
        mask=labels==c;center=z[mask].mean(0);centers.append(center)
        for d in np.unique(speeds[mask]):within.append(float(np.square(z[mask&(speeds==d)].mean(0)-center).sum()))
    centers=np.stack(centers);between=float(np.mean([np.square(centers[i]-centers[j]).sum() for i in range(len(centers)) for j in range(i)]))
    return {'within_class_speed_center_variance':float(np.mean(within)),'between_class_center_distance':between,'ratio':float(np.mean(within)/max(between,1e-12)),'sample_feature_variance_trace':float(z.var(0).sum())}

def evaluate_task(model,source,wave,rows,y,role,memory,support,cfg):
    val=record_prefix(rows,role,'base_val',cfg.validation_windows_per_csv);temperature=record_prefix(rows,role,'temperature_val',cfg.validation_windows_per_csv);query=np.flatnonzero(role=='query')
    novel=support['indices'][0]+support['indices'][1];ids=np.unique(np.r_[memory,novel,temperature,query,val]);lookup={int(i):j for j,i in enumerate(ids)}
    values=extract(model,wave,ids,cfg.device,cfg.batch_size);ref=np.array([lookup[int(i)] for i in memory]);new=[[lookup[int(i)] for i in group] for group in support['indices']]
    q=np.array([lookup[int(i)] for i in query]);t=np.array([lookup[int(i)] for i in temperature]);v=np.array([lookup[int(i)] for i in val])
    logits=classify(values,ref,q,y[memory],cfg,new);tlogits=classify(values,ref,t,y[memory],cfg,new)
    scale=fit_temperature(tlogits[None],y[temperature]);raw=softmax(logits,axis=-1);p=softmax(logits/scale,axis=-1)
    fp=expit(values['d_logits'][q]);truth=SIGNATURES[y[query]];component=[]
    for j in range(4):
        a=truth[:,j]>.5;b=fp[:,j]>=.5;tp=int((a&b).sum());false=int((~a&b).sum());fn=int((a&~b).sum())
        component.append({'recall':float(b[a].mean()) if a.any() else None,'f1':2*tp/max(2*tp+false+fn,1),'false_positive_rate':float(b[~a].mean()) if (~a).any() else None})
    before=extract(source,wave,memory,cfg.device,cfg.batch_size);drift={key:float(np.mean(1-np.sum(values[key][ref]*before[key],axis=-1))) for key in ('d','s') if key in values and key in before}
    stability={key:stability_diagnostic(values[key][v],y[val],np.array([rows[i]['speed_setting_hz'] for i in val])) for key in ('d','s') if key in values}
    report={'raw':metrics(y[query],raw),'calibrated':metrics(y[query],p),'temperature':scale,'components':component,'replay_feature_drift':drift,'source_val_stability':stability,'mean_gate':values['gate']}
    return report,{'probabilities':p,'raw_probabilities':raw,'component_probabilities':fp,'true_labels':y[query],'query_indices':query}

def summarize(root):
    items=[]
    for path in sorted(root.glob('*/tasks/*.json')):
        item=json.loads(path.read_text(encoding='utf-8'))
        if item.get('finished'):items.append(item)
    summary=[]
    for mode in MODES:
        for k in sorted({v['k'] for v in items}):
            group=[v for v in items if v['mode']==mode and v['k']==k]
            if not group:continue
            summary.append({'mode':mode,'k':k,'tasks':len(group),**{key:float(np.mean([v['report']['raw'][key] for v in group])) for key in ('accuracy','macro_f1','base_f1','novel_f1','ece')},'calibrated_ece':float(np.mean([v['report']['calibrated']['ece'] for v in group])),'inner_component_f1':float(np.mean([v['report']['components'][2]['f1'] for v in group])),'outer_component_f1':float(np.mean([v['report']['components'][3]['f1'] for v in group])),'stable_change':float(np.mean([v['update_audit']['stable_max_change'] for v in group]))})
    dump(root/'metrics.json',items);dump(root/'summary.json',summary)
    if summary:csv_out(root/'summary.csv',summary)
    return items

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'lwc_stability_pilot.json');parser.add_argument('--check',action='store_true');parser.add_argument('--resume',type=Path);args=parser.parse_args()
    settings=json.loads(args.config.read_text(encoding='utf-8'));cfg=SimpleNamespace(**settings)
    if cfg.shots!=[1,5,10] or cfg.wavelet_kernel%2!=1 or cfg.feature_pool<1 or cfg.gate_max<=0:raise ValueError('shot/LWC/门控配置错误')
    rows,y,_=load_samples(ROOT/cfg.reference_cache,metadata_only=True)
    for case in cfg.cases:
        for seed in cfg.seeds:
            role,*_=split_cross(rows,case,seed);role,_=reserve_calibration(rows,y,role,seed);memory=fixed_memory(rows,y,role,seed,cfg.memory_per_base_class)
            sample_cross_speed(np.flatnonzero(role=='base_train'),rows,y,np.random.default_rng(seed),cfg.source_per_domain);sample_cross_speed(memory,rows,y,np.random.default_rng(seed),cfg.replay_per_domain)
            for k in cfg.shots:cross_supports(rows,y,role,case,seed,k,1)
    cfg.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('LWC四组预检查通过；设备='+str(cfg.device),flush=True)
    if args.check:return
    torch.set_num_threads(2);started=time.perf_counter()
    if args.resume:
        root=args.resume.resolve()
        if json.loads((root/'config.json').read_text(encoding='utf-8'))!=settings:raise ValueError('续跑配置必须相同')
    else:
        root=ROOT/cfg.output;original=root;n=2
        while root.exists():root=original.with_name(original.name+f'_run{n}');n+=1
        root.mkdir(parents=True);dump(root/'config.json',settings);snapshot=root/'source_snapshot';snapshot.mkdir()
        for name in ('lwc_stability_model.py','run_lwc_stability.py','show_lwc_stability.py','mahalanobis_metric.py','run_laspi_cross_702010.py','temporal_information.py','run_dggn_laspi.py'):shutil.copy2(ROOT/name,snapshot/name)
    dump(root/'execution.json',{'finished':False,'device':str(cfg.device)});print('RESULT_DIR='+str(root),flush=True)
    wave=prepare_waveforms(cfg,rows)
    for case in cfg.cases:
        for seed in cfg.seeds:
            role,_,_,_,audit=split_cross(rows,case,seed);role,reserved=reserve_calibration(rows,y,role,seed);memory=fixed_memory(rows,y,role,seed,cfg.memory_per_base_class)
            folder=root/f'{case}_s{seed}';folder.mkdir(exist_ok=True);(folder/'tasks').mkdir(exist_ok=True)
            csv_out(folder/'split.csv',[{'sample_id':r['sample_id'],'record_id':r['record_id'],'role':role[i]} for i,r in enumerate(rows)]);dump(folder/'split_audit.json',{**audit,'reserved_validation_csvs':reserved,'memory_indices':memory.tolist()})
            seed_all(seed);initial=DualLWC(cfg,'parallel').to(cfg.device);initial=train_source(initial,wave,rows,y,role,memory,cfg,seed,folder/'common_pretrain','pretrain')
            sources={}
            for mode in ('single','parallel','frozen_exchange'):
                model=copy.deepcopy(initial);model.mode=mode
                sources[mode]=train_source(model,wave,rows,y,role,memory,cfg,seed,folder/('source_'+mode),'source')
            sources['trainable_exchange']=copy.deepcopy(sources['frozen_exchange']);sources['trainable_exchange'].mode='trainable_exchange'
            if any(not torch.equal(v,sources['trainable_exchange'].state_dict()[k]) for k,v in sources['frozen_exchange'].state_dict().items()):raise AssertionError('第3/4组源起点不一致')
            for k in cfg.shots:
                for support in tqdm(cross_supports(rows,y,role,case,seed,k,cfg.support_repeats),desc=f'四组配对 {case} s{seed} {k}-shot',unit='支持集',dynamic_ncols=True):
                    for mode in MODES:
                        taskpath=folder/'tasks'/f'{mode}_k{k}_e{support["episode"]}.json'
                        if taskpath.exists():continue
                        source=sources[mode];original_state={n:v.detach().cpu().clone() for n,v in source.state_dict().items()}
                        model,history,changes=adapt(source,wave,rows,y,memory,support,cfg,seed*100000+k*100+support['episode'],mode)
                        if state_change(original_state,source)!=0:raise AssertionError('源模型被适配修改')
                        report,predictions=evaluate_task(model,source,wave,rows,y,role,memory,support,cfg)
                        result={'finished':True,'mode':mode,'condition':case,'seed':seed,'k':k,'episode':support['episode'],'support_sample_ids':support['sample_ids'],'report':report,'update_audit':changes,'source_unchanged':True}
                        if support['episode']==0:
                            output=folder/mode;output.mkdir(exist_ok=True);dump(output/f'history_{k}shot.json',history);save_predictions(output/f'predictions_{k}shot.npz',**predictions);save_checkpoint(model.state_dict(),output/f'adapted_{k}shot.pt')
                        dump(taskpath,result)
                        print(f'{mode} {k}-shot #{support["episode"]+1}: MacroF1={report["raw"]["macro_f1"]:.3f}, 基类={report["raw"]["base_f1"]:.3f}, 复合={report["raw"]["novel_f1"]:.3f}, 稳定漂移={changes["stable_max_change"]:.3g}',flush=True)
                        del model
            del sources,initial
            summarize(root)
    items=summarize(root);expected=len(cfg.cases)*len(cfg.seeds)*len(cfg.shots)*cfg.support_repeats*4
    if len(items)!=expected:raise AssertionError(f'任务数不完整：{len(items)}/{expected}')
    dump(root/'execution.json',{'finished':True,'device':str(cfg.device),'tasks':len(items),'elapsed_seconds':time.perf_counter()-started})
    if cfg.show_plots:
        from show_lwc_stability import show
        show(root)

if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    main()
