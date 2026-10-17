"""四成分BCE监督＋BNN＋成分门控组合原型。填写component_train_config.json运行。

源训练只有五基类；每任务仅指定K复合支持标签可更新成分层。
网络输出不是7类softmax；7类诊断由成分签名及马氏组合原型形成。
"""
import argparse, copy, json, os, shutil, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax, expit
from prepare_laspi import ROOT, CN, dump, csv_out
from run_laspi_gfsl import Encoder, batch, load_samples, seed_all, save_checkpoint, save_predictions, fit_temperature, metrics
from run_laspi_vibration_order import vibration_images, scaler_for_vibration, mixed_episode, episode_reference, state_cpu
from run_laspi_cross_702010 import split_cross, cross_supports, CASES
from bayesian_layers import FeatureHead, BayesianLinear
from mahalanobis_metric import fit_reference, whiten, mahalanobis_logits
from run_laspi_compositional import compose, distance_logits, COMPONENTS

# 顺序：齿面损伤、半齿断裂、轴承内圈、轴承外圈。
SIGNATURES=np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,1,0],[0,1,0,1]],np.float32)
FACTOR_NAMES=('齿面损伤','半齿断裂','内圈','外圈')
SCHEMES=('support_only','fixed_composition','component_gated')


class ComponentNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if getattr(cfg,'encoder_kind','legacy')=='temporal_audit':
            from temporal_information import DiagnosticEncoder
            self.encoder=DiagnosticEncoder(cfg.preserve_time)
        else:self.encoder=Encoder(1)
        self.encoder.classifier=nn.Identity()
        self.features=FeatureHead(cfg.bayesian,cfg.prior_std)
        self.components=BayesianLinear(32,4,cfg.bayesian,cfg.prior_std)
    def resample(self):self.features.resample();self.components.resample()
    def forward(self,x):
        z=self.features(self.encoder(x));return z,self.components(z)
    def kl(self):return self.features.kl()+self.components.kl()
    def count(self):return self.features.count()+self.components.weight_mu.numel()+self.components.bias_mu.numel()


@torch.no_grad()
def draw(model,images,ids,scaler,device,count):
    model.eval();features=[];logits=[]
    for _ in range(count):
        model.resample();zs=[];ls=[]
        # 一次后验抽样在全部批次共享，不能每个窗口单独改权重。
        for begin in range(0,len(ids),64):
            z,l=model(batch(images,ids[begin:begin+64],[0],scaler,device))
            zs.append(z.cpu().numpy());ls.append(l.cpu().numpy())
        features.append(np.concatenate(zs));logits.append(np.concatenate(ls))
    return np.stack(features),np.stack(logits)


def signature_logits(logits):
    # 独立Bernoulli工作假设下，计算七种成分签名的log likelihood。
    return np.einsum('mnf,cf->mnc',-np.logaddexp(0,-logits),SIGNATURES)+np.einsum('mnf,cf->mnc',-np.logaddexp(0,logits),1-SIGNATURES)


def train_source(images,rows,y,role,cond,conditions,scaler,cfg,seed,folder,device):
    seed_all(seed);model=ComponentNet(cfg).to(device)
    groups=[{'params':model.encoder.parameters(),'lr':cfg.encoder_lr},
            {'params':list(model.features.parameters())+list(model.components.parameters()),'lr':cfg.head_lr}]
    optimizer=torch.optim.Adam(groups);rng=np.random.default_rng(seed)
    tr=np.flatnonzero(role=='base_train');va=np.flatnonzero(role=='base_val')
    target=torch.as_tensor(SIGNATURES,device=device);bank_labels=torch.as_tensor(y[tr],device=device)
    best=float('inf');history=[]
    epochs=tqdm(range(1,cfg.epochs+1),desc=f"源CNN {folder.name}",unit="轮",dynamic_ncols=True,mininterval=1)
    for epoch in epochs:
        # 仅源训练池。池每轮刷新，排除当前episode所有窗口。
        bank=None
        if cfg.metric_loss_weight:
            values,_=draw(model,images,tr,scaler,device,1);bank=torch.as_tensor(values[0],device=device)
        model.train();losses=[];bces=[];counts={}
        for _ in tqdm(range(cfg.steps_per_epoch),desc="训练批次",unit="批",leave=False,position=1,dynamic_ncols=True,mininterval=1):
            ids,support,query,k=mixed_episode(y,role,cond,conditions,rng,cfg.train_shots)
            yy=torch.as_tensor(y[ids],device=device);optimizer.zero_grad(set_to_none=True)
            loss=0.;bce_value=0.;draws=cfg.mc_train if cfg.bayesian else 1
            for _ in range(draws):
                model.resample();z,l=model(batch(images,ids,[0],scaler,device))
                bce=F.binary_cross_entropy_with_logits(l,target[yy]);term=cfg.component_loss_weight*bce
                if cfg.metric_loss_weight:
                    keep=torch.as_tensor(~np.isin(tr,ids),device=device)
                    means,chol=episode_reference(z,yy,support,bank,bank_labels,keep,cfg.shrinkage)
                    term+=cfg.metric_loss_weight*F.cross_entropy(mahalanobis_logits(z[query],means,chol),yy[query])
                loss+=term;bce_value+=float(bce.detach())
            loss=loss/draws+cfg.kl_beta*model.kl()/model.count()
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()));bces.append(bce_value/draws);counts[k]=counts.get(k,0)+1
        _,vl=draw(model,images,va,scaler,device,cfg.mc_val if cfg.bayesian else 1)
        truth=SIGNATURES[y[va]][None]
        val_bce=float((truth*np.logaddexp(0,-vl)+(1-truth)*np.logaddexp(0,vl)).mean())
        fp=expit(vl).mean(0);factor_exact=float(((fp>=.5)==SIGNATURES[y[va]]).all(-1).mean())
        base_probability=softmax(signature_logits(vl)[:,:,:5],axis=-1).mean(0)
        accuracy=float((base_probability.argmax(-1)==y[va]).mean())
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'component_bce':float(np.mean(bces)),
                        'val_component_bce':val_bce,'val_factor_exact':factor_exact,'val_base_signature_accuracy':accuracy,'shot_counts':counts})
        if val_bce<best:best=val_bce;save_checkpoint(state_cpu(model),folder/'best_components.pt')
        epochs.set_postfix(loss=f'{np.mean(losses):.4f}',val_BCE=f'{val_bce:.4f}')
        dump(folder/'training_history.json',history)
        print(f'训练 {folder.name} {epoch}/{cfg.epochs}：总损失={np.mean(losses):.4f} 成分BCE={np.mean(bces):.4f} 源验证BCE={val_bce:.4f} 四成分全对={factor_exact:.2%} 五基类签名Acc={accuracy:.2%} 支持数={counts}',flush=True)
    model.load_state_dict(torch.load(folder/'best_components.pt',map_location=device,weights_only=True))
    return model


def adapt_components(model,z,y,role,support,cfg,seed,device):
    # 每个支持任务从同一源模型重置；仅更新成分输出层，不更新编码器/特征头。
    layer=BayesianLinear(32,4,cfg.bayesian,cfg.prior_std).to(device)
    layer.load_state_dict(state_cpu(model.components));rng=np.random.default_rng(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    x=torch.as_tensor(z.mean(0),device=device);targets=torch.as_tensor(SIGNATURES[y],device=device)
    novel=np.array(support['indices'][0]+support['indices'][1]);optimizer=torch.optim.Adam(layer.parameters(),lr=cfg.adapt_lr)
    for _ in range(cfg.adapt_steps):
        replay=np.concatenate([rng.choice(np.flatnonzero((role=='base_train')&(y==c)),cfg.replay_per_class,replace=False) for c in range(5)])
        optimizer.zero_grad(set_to_none=True);loss=0.;draws=cfg.mc_train if cfg.bayesian else 1
        for _ in range(draws):
            layer.resample()
            # 基类回放与少样本复合支持各占一半，避免K变化改变任务权重。
            loss+=.5*F.binary_cross_entropy_with_logits(layer(x[replay]),targets[replay])+.5*F.binary_cross_entropy_with_logits(layer(x[novel]),targets[novel])
        loss=loss/draws+cfg.kl_beta*layer.kl()/(layer.weight_mu.numel()+layer.bias_mu.numel())
        loss.backward();nn.utils.clip_grad_norm_(layer.parameters(),5);optimizer.step()
    logits=[]
    with torch.no_grad():
        for values in z:
            layer.resample();logits.append(layer(torch.as_tensor(values,device=device)).cpu().numpy())
    return np.stack(logits),state_cpu(layer)


def evaluate(model,z,rows,y,role,target,seed,cfg,result,device):
    tr=np.flatnonzero(role=='base_train');va=np.flatnonzero(role=='base_val');qu=np.flatnonzero(role=='query')
    w=[];base=[]
    with torch.no_grad():
        for values in z:
            x=torch.as_tensor(values);means,chol=fit_reference(x[tr],torch.as_tensor(y[tr]),shrinkage=cfg.shrinkage)
            w.append(whiten(x,chol).numpy());base.append(whiten(means,chol).numpy())
    w=np.stack(w).astype(np.float64);base=np.stack(base).astype(np.float64);records=[]
    predictions={s:[] for s in SCHEMES};factor_predictions=[];shot_ids=[];alphas=[];support_log=[];states=[]
    for k in cfg.shots:
        for support in cross_supports(rows,y,role,target,seed,k,cfg.support_repeats):
            logits,state=adapt_components(model,z,y,role,support,cfg,seed*100000+k*100+support['episode'],device)
            states.append(state);fp=expit(logits);factor_scores=signature_logits(logits);factor_predictions.append(fp[:,qu].mean(0))
            support_log.append(support);shot_ids.append(k);gates=[]
            for c,ids in zip((5,6),support['indices']):
                a,b=COMPONENTS[c];f1=int(np.flatnonzero(SIGNATURES[a])[0]);f2=int(np.flatnonzero(SIGNATURES[b])[0])
                evidence=fp[:,ids].mean(1)
                gates.append(evidence[:,f1]/np.maximum(evidence[:,f1]+evidence[:,f2],1e-8))
            alphas.append(np.stack(gates,axis=-1))
            for scheme in SCHEMES:
                novel=[]
                for j,(c,ids) in enumerate(zip((5,6),support['indices'])):
                    if scheme=='component_gated':
                        a,b=COMPONENTS[c];alpha=gates[j][:,None];prior=alpha*base[:,a]+(1-alpha)*base[:,b]
                        weight=k/(k+cfg.prior_strength);p=(1-weight)*prior+weight*w[:,ids].mean(1)
                    else:p,_,weight=compose(base,w[:,ids],COMPONENTS[c],scheme,cfg.prior_strength)
                    novel.append(p[:,None])
                means=np.concatenate([base]+novel,axis=1)
                vl=distance_logits(w[:,va],means)+cfg.factor_score_weight*factor_scores[:,va]
                ql=distance_logits(w[:,qu],means)+cfg.factor_score_weight*factor_scores[:,qu]
                temperature=fit_temperature(vl,y[va]);prob=softmax(ql/temperature,axis=-1).mean(0)
                truth=SIGNATURES[y[qu]];component_prob=fp[:,qu].mean(0)
                records.append({'condition':target,'seed':seed,'k':k,'episode':support['episode'],'scheme':scheme,
                                'calibrated':metrics(y[qu],prob),'raw':metrics(y[qu],softmax(ql,axis=-1).mean(0)),
                                'temperature':temperature,'support_weight':weight,'alpha_mean':np.mean(gates,axis=1).tolist(),
                                'factor_exact_accuracy':float(((component_prob>=.5)==truth).all(-1).mean()),
                                'novel_factor_exact':float(((component_prob[y[qu]>=5]>=.5)==truth[y[qu]>=5]).all(-1).mean())})
                predictions[scheme].append(prob)
            print(f'支持适配 {target} s{seed} {k}-shot {support["episode"]+1}/{cfg.support_repeats}：仅成分层更新，alpha={np.round(np.mean(gates,axis=1),3).tolist()}',flush=True)
        for scheme in SCHEMES:
            v=[r for r in records if r['k']==k and r['scheme']==scheme]
            print(f'测试 {target} {k}-shot {scheme}：MacroF1={np.mean([r["calibrated"]["macro_f1"] for r in v]):.4f} 基类F1={np.mean([r["calibrated"]["base_f1"] for r in v]):.4f} 复合F1={np.mean([r["calibrated"]["novel_f1"] for r in v]):.4f}',flush=True)
    prefix=f'{target}_s{seed}'
    dump(result/'predictions'/f'{prefix}_supports.json',support_log);save_checkpoint(states,result/'predictions'/f'{prefix}_adapted_component_states.pt')
    for scheme in SCHEMES:
        save_predictions(result/'predictions'/f'{prefix}_{scheme}.npz',probabilities=np.stack(predictions[scheme]),true_labels=y[qu],query_indices=qu,
                         factor_probabilities=np.stack(factor_predictions),component_alpha=np.stack(alphas),shots=np.array(shot_ids))
    return records


def summary(result,records):
    rows=[]
    for scheme in SCHEMES:
        for k in sorted(set(r['k'] for r in records)):
            subset=[r for r in records if r['k']==k and r['scheme']==scheme]
            rows.append({'scheme':scheme,'k':k,**{name:float(np.mean([r['calibrated'][name] for r in subset])) for name in ('accuracy','macro_f1','base_f1','novel_f1','ece')},
                         'novel_factor_exact':float(np.mean([r['novel_factor_exact'] for r in subset]))})
    dump(result/'summary.json',rows);csv_out(result/'summary.csv',rows)
    return rows


def show(result):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib_cache'))
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    records=json.loads((result/'summary.json').read_text(encoding='utf-8'));fig,axs=plt.subplots(1,3,figsize=(16,5))
    fig.canvas.manager.set_window_title('成分BCE训练：普通与组合原型对照')
    for ax,key,title in zip(axs,('macro_f1','base_f1','novel_f1'),('七类Macro-F1','基类F1','复合F1')):
        for scheme,label in [('support_only','普通支持原型'),('fixed_composition','固定成分组合'),('component_gated','网络成分门控')]:
            ss=[r for r in records if r['scheme']==scheme];ax.plot([r['k'] for r in ss],[r[key] for r in ss],marker='o',label=label)
        ax.set(title=title,xlabel='复合类支持窗口数',ylim=(0,1));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    config=json.loads((result/'config.json').read_text(encoding='utf-8'));case=config['cases'][0];seed=config['seeds'][0]
    history=json.loads((result/'models'/f'{case}_s{seed}'/'training_history.json').read_text(encoding='utf-8'))
    fig2,axes=plt.subplots(1,2,figsize=(12,5));fig2.canvas.manager.set_window_title('成分网络训练与复合成分概率')
    for key,label in [('component_bce','训练BCE'),('val_component_bce','源验证BCE')]:axes[0].plot([r['epoch'] for r in history],[r[key] for r in history],label=label)
    axes[0].set(title='成分监督训练',xlabel='轮数');axes[0].legend()
    with np.load(result/'predictions'/f'{case}_s{seed}_component_gated.npz') as data:
        fp=data['factor_probabilities'][0];y=data['true_labels'];heat=np.stack([fp[y==c].mean(0) for c in range(7)])
        axes[1].imshow(heat,vmin=0,vmax=1,cmap='Blues');axes[1].set_xticks(range(4),FACTOR_NAMES);axes[1].set_yticks(range(7),CN);axes[1].set_title('固定首支持任务：各类成分概率均值')
    fig2.tight_layout();plt.show()


def validate_config(d):
    if not set(d['cases'])<=set(CASES) or not d['cases'] or not d['seeds']:raise ValueError('工况或种子无效')
    if d['axis'] not in ('hz','order') or not isinstance(d['bayesian'],bool):raise ValueError('axis须为hz/order，bayesian须为布尔值')
    for key in ('epochs','steps_per_epoch','support_repeats','mc_train','mc_val','mc_eval','replay_per_class'):
        if not isinstance(d[key],int) or d[key]<1:raise ValueError(key+'须为正整数')
    if not isinstance(d['adapt_steps'],int) or d['adapt_steps']<0:raise ValueError('adapt_steps须为非负整数')
    if not d['shots'] or not d['train_shots'] or min(d['shots']+d['train_shots'])<1 or max(d['train_shots'])>24:raise ValueError('shot无效，训练shot最大24')
    for key in ('encoder_lr','head_lr','adapt_lr','prior_std','order_max','component_loss_weight'):
        if d[key]<=0:raise ValueError(key+'须大于0')
    for key in ('metric_loss_weight','kl_beta','prior_strength','factor_score_weight'):
        if d[key]<0:raise ValueError(key+'须非负')
    if not 0<d['shrinkage']<=1:raise ValueError('shrinkage须在(0,1]')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'component_train_config.json');ap.add_argument('--no-show',action='store_true');ap.add_argument('--reuse-source',type=Path,help='核对训练参数和CSV分区后复用已完成源模型');args=ap.parse_args()
    settings=json.loads(args.config.read_text(encoding='utf-8'));validate_config(settings);cfg=SimpleNamespace(**settings)
    result=Path(cfg.output);result=result if result.is_absolute() else ROOT/result
    original=result;number=2
    while result.exists():result=original.with_name(original.name+f'_run{number}');number+=1
    for name in ('models','predictions','splits','source_snapshot'):(result/name).mkdir(parents=True,exist_ok=True)
    dump(result/'config.json',settings)
    reuse_config=None
    if args.reuse_source:
        reuse_config=json.loads((args.reuse_source/'config.json').read_text(encoding='utf-8'))
        keys=('axis','order_max','bayesian','epochs','steps_per_epoch','encoder_lr','head_lr','train_shots','component_loss_weight','metric_loss_weight','kl_beta','prior_std','shrinkage','mc_train','mc_val','mc_eval')
        if any(reuse_config.get(key)!=settings[key] for key in keys):raise ValueError('已完成源模型与本次训练参数不一致，不能复用')
    for name in ('train_laspi_components.py','bayesian_layers.py','run_laspi_vibration_order.py','run_laspi_compositional.py','run_laspi_cross_702010.py','mahalanobis_metric.py','run_laspi_gfsl.py','prepare_laspi.py'):shutil.copy2(ROOT/name,result/'source_snapshot'/name)
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');started=time.perf_counter()
    print(f'设备={device}；GPU={torch.cuda.get_device_name(0) if device.type=="cuda" else "无"}；四成分={FACTOR_NAMES}；结果={result}',flush=True)
    rows,y,raw=load_samples(ROOT/'prepared_window_zscore',mmap=False);images,_=vibration_images(raw,rows,cfg.axis,cfg.order_max);del raw
    records=[]
    for case in cfg.cases:
        for seed in cfg.seeds:
            role,partition,cond,conditions,audit=split_cross(rows,case,seed)
            for c in (5,6):
                count=len({rows[i]['record_id'] for i in np.flatnonzero((role=='novel_support_pool')&(y==c))})
                if max(cfg.shots)>count:raise ValueError(f'类别{c}只有{count}个支持CSV')
            dump(result/'splits'/f'{case}_s{seed}.json',audit)
            csv_out(result/'splits'/f'{case}_s{seed}.csv',[{'sample_id':r['sample_id'],'record_id':r['record_id'],'role':role[i],'partition':partition[i],'label':int(y[i])} for i,r in enumerate(rows)])
            folder=result/'models'/f'{case}_s{seed}';folder.mkdir(exist_ok=True)
            scaler=scaler_for_vibration(images,role=='base_train');dump(folder/'image_scaler.json',scaler)
            old_folder=args.reuse_source/'models'/f'{case}_s{seed}' if args.reuse_source else None
            if old_folder and (old_folder/'posterior_features.npy').exists() and (old_folder/'best_components.pt').exists():
                import csv
                with (args.reuse_source/'splits'/f'{case}_s{seed}.csv').open(encoding='utf-8-sig') as stream:old_split=list(csv.DictReader(stream))
                if [(r['sample_id'],r['role']) for r in old_split]!=[(r['sample_id'],role[i]) for i,r in enumerate(rows)]:raise ValueError('复用源模型CSV分区不一致')
                old_scaler=json.loads((old_folder/'image_scaler.json').read_text(encoding='utf-8'))
                if old_scaler!=scaler:raise ValueError('源归一化统计不一致')
                model=ComponentNet(cfg).to(device);model.load_state_dict(torch.load(old_folder/'best_components.pt',map_location=device,weights_only=True))
                for file in ('best_components.pt','training_history.json'):shutil.copy2(old_folder/file,folder/file)
                with (old_folder/'posterior_features.npy').open('rb') as stream:z=np.load(stream,allow_pickle=False)
                dump(folder/'reuse_audit.json',{'source':str(old_folder),'training_parameters_match':True,'split_match':True,'normalizer_match':True})
                print(f'复用已完成60轮源模型：{old_folder}；参数、CSV分区和标准化核验一致。',flush=True)
            else:
                model=train_source(images,rows,y,role,cond,conditions,scaler,cfg,seed,folder,device)
                z,_=draw(model,images,np.arange(len(rows)),scaler,device,cfg.mc_eval if cfg.bayesian else 1)
            # 固定特征后验抽样和任务索引以便回放，不在支持任务中重新训练特征。
            with (folder/'posterior_features.npy').open('wb') as stream:np.save(stream,z)
            records.extend(evaluate(model,z,rows,y,role,case,seed,cfg,result,device));dump(result/'metrics.json',records);summary(result,records)
    dump(result/'execution.json',{'finished':True,'elapsed_seconds':time.perf_counter()-started,'device':str(device)})
    print('训练与评价完成：'+str(result),flush=True)
    if cfg.show_plots and not args.no_show:show(result)
    return result


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    main()
