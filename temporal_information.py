"""Controlled input/encoder ablations. Original data and existing caches are read-only."""
import collections,csv,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import resample_poly
from tqdm.auto import tqdm
from prepare_laspi import ROOT,inventory,sha256,dump,csv_out,normalize_windows
from run_laspi_gfsl import batch

VARIANTS={
    'legacy':{'fs':12800,'preserve_time':False,'normalization':'window_zscore'},
    'fullrate':{'fs':25600,'preserve_time':False,'normalization':'window_zscore'},
    'fullrate_temporal':{'fs':25600,'preserve_time':True,'normalization':'window_zscore'},
    'fullrate_temporal_amplitude':{'fs':25600,'preserve_time':True,'normalization':'none'},
}

class DiagnosticEncoder(nn.Module):
    """Same parameter count/project/pooling; only temporal strides change."""
    def __init__(self,preserve_time=True):
        super().__init__();layers=[]
        for j,(a,b) in enumerate(zip((1,16,32,64),(16,32,64,128))):
            stride=(2,2 if j==0 or not preserve_time else 1)
            layers += [nn.Conv2d(a,b,5 if j==0 else 3,stride=stride,padding=2 if j==0 else 1,bias=False),nn.GroupNorm(4 if b==16 else 8,b),nn.SiLU()]
        self.body=nn.Sequential(*layers)
        self.project=nn.Sequential(nn.Flatten(),nn.Linear(1024,128),nn.LayerNorm(128))
    def forward(self,x):
        x=self.body(x)
        p=torch.stack([v.mean((2,3)) for v in torch.tensor_split(x,8,dim=2)],dim=2)[:,:,:,None]
        return F.normalize(self.project(p),dim=-1)

def prepare_cache(cfg,variant):
    spec=VARIANTS[variant];folder=ROOT/cfg.cache_root/f"fs{spec['fs']}_{spec['normalization']}"
    contract={'fs':spec['fs'],'length':int(spec['fs']*.32),'nfft':int(spec['fs']*.08),'hop':int(spec['fs']*.02),'normalization':spec['normalization'],'source':str(Path(cfg.raw_source).resolve()),'format_version':1}
    marker=folder/'complete.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8'))['contract']!=contract:raise ValueError('缓存参数不匹配：'+str(folder))
        return folder
    if folder.exists():raise FileExistsError('保留未完成缓存，请改cache_root或人工检查：'+str(folder))
    records=inventory(Path(cfg.raw_source));expected={r['record_id']:r for r in csv.DictReader((ROOT/cfg.reference_cache/'records.csv').open(encoding='utf-8-sig'))}
    folder.mkdir(parents=True);nfft=contract['nfft'];length=contract['length'];hop=contract['hop']
    images=np.lib.format.open_memmap(folder/'stft.npy',mode='w+',dtype=np.float16,shape=(len(records)*31,1,nfft//2+1,13))
    rows=[];audits=[];hann=torch.hann_window(nfft)
    for j,r in enumerate(tqdm(records,desc='原始信号与频带审计',unit='CSV',dynamic_ncols=True)):
        digest=sha256(r['file'])
        if digest!=expected[r['record_id']]['sha256']:raise ValueError('原始文件已变化：'+r['file'])
        raw=np.loadtxt(r['file'],delimiter=',',dtype=np.float32,usecols=[3])
        if raw.shape!=(256000,) or not np.isfinite(raw).all():raise ValueError('振动通道异常：'+r['file'])
        raw_chunks=raw[:31*8192].reshape(31,8192);raw_rms=np.sqrt(np.mean((raw_chunks-raw_chunks.mean(1,keepdims=True))**2,axis=1))
        spectrum=np.abs(np.fft.rfft(raw-raw.mean()))**2;frequency=np.fft.rfftfreq(len(raw),1/25600)
        # One-sided energy bookkeeping, including DC/Nyquist endpoints.
        spectrum[1:-1]*=2
        high=float(spectrum[frequency>6400].sum()/max(spectrum.sum(),1e-20))
        signal=raw if spec['fs']==25600 else resample_poly(raw,1,2).astype(np.float32)
        chunks=signal[:31*length].reshape(31,length)
        normalized=normalize_windows(chunks,spec['normalization'])
        ft=torch.stft(torch.from_numpy(normalized),n_fft=nfft,hop_length=hop,win_length=nfft,window=hann,center=False,normalized=True,return_complex=True)
        image=torch.log1p(ft.abs().square()).numpy()
        if image.shape!=(31,nfft//2+1,13) or not np.isfinite(image).all():raise ValueError('时频形状/数值异常')
        # Float16 storage errors do not establish diagnostic sufficiency.
        err=float(np.linalg.norm(image.astype(np.float16).astype(np.float32)-image)/max(np.linalg.norm(image),1e-12))
        images[j*31:(j+1)*31,0]=image.astype(np.float16);r['sha256']=digest;r['windows']=31
        for w in range(31):
            rows.append({**{k:r[k] for k in ('record_id','condition_group','label','class_name','speed_setting_hz','load_percent','motor_rpm_filename')},'sample_id':r['record_id']+f'_w{w:03d}','window_index':w,'cache_index':j*31+w,'start_s':w*.32,'duration_s':.32})
        audits.append({'record_id':r['record_id'],'label':r['label'],'condition':f"{r['speed_setting_hz']}hz_{r['load_percent']}pct",'raw_highband_energy_ratio':high,'raw_window_rms_mean':float(raw_rms.mean()),'raw_window_rms_cv':float(raw_rms.std()/max(raw_rms.mean(),1e-12)),'normalized_rms_mean':float(np.sqrt(np.mean(normalized**2,axis=1)).mean()),'float16_relative_error':err})
    images.flush();del images
    csv_out(folder/'records.csv',records);csv_out(folder/'samples.csv',rows);csv_out(folder/'preprocessing_audit.csv',audits)
    dump(marker,{'contract':contract,'windows_per_record':31,'duration_s':.32,'analysis_seconds':.08,'step_seconds':.02,'frequency_bin_spacing_hz':12.5,'phase_preserved':False,'tail_discarded_seconds':.08,'physical_units_calibrated':False})
    return folder

class BoundedPrefix:
    """On-demand frozen prefix with bounded RAM LRU; no multi-GB disk cache."""
    def __init__(self,encoder,images,scaler,device,batch_size=32,budget_mb=192):
        self.prefix=encoder.body[:9].eval();self.images=images;self.scaler=scaler;self.device=device;self.batch_size=batch_size
        self.cache=collections.OrderedDict();self.bytes=0;self.budget=budget_mb*2**20
        for p in self.prefix.parameters():p.requires_grad_(False)
    @torch.no_grad()
    def __getitem__(self,indices):
        indices=np.asarray(indices,dtype=int).reshape(-1);missing=list(dict.fromkeys(int(i) for i in indices if int(i) not in self.cache))
        # Keep references for this request: LRU insertion may evict a later requested item.
        computed={int(i):self.cache[int(i)] for i in indices if int(i) in self.cache}
        for b in range(0,len(missing),self.batch_size):
            ids=missing[b:b+self.batch_size];values=self.prefix(batch(self.images,ids,[0],self.scaler,self.device)).cpu().numpy()
            for i,v in zip(ids,values):computed[i]=v.copy()
        output=[]
        for i in indices:
            i=int(i)
            v=computed[i]
            if i in self.cache:old=self.cache.pop(i);self.bytes-=old.nbytes
            output.append(v)
            if v.nbytes<=self.budget:
                while self.cache and self.bytes+v.nbytes>self.budget:
                    _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
                self.cache[i]=v;self.bytes+=v.nbytes
        return np.stack(output)

def reserve_calibration(rows,y,role,seed):
    """Three disjoint subsets of source-base validation CSVs; no novel labels."""
    role=role.copy();rng=np.random.default_rng(seed+1551);audit={}
    for c in range(5):
        files=sorted({rows[i]['record_id'] for i in np.flatnonzero((role=='base_val')&(y==c))})
        if len(files)<3:raise ValueError('基类验证CSV不足以分离选模/温度/停止阈值')
        files=list(rng.permutation(files));groups=[files[k::3] for k in range(3)]
        for name,names in zip(('base_val','temperature_val','policy_val'),groups):
            ids=set(names)
            for i,r in enumerate(rows):
                if r['record_id'] in ids:role[i]=name
            audit.setdefault(name,[]).extend(sorted(ids))
    assert not any(set(audit[a])&set(audit[b]) for a,b in [('base_val','temperature_val'),('base_val','policy_val'),('temperature_val','policy_val')])
    return role,audit

@torch.no_grad()
def probe_information(encoder,images,scaler,rows,y,role,device,folder,seed,max_per_class=64):
    """Matched 64-D random-projection linear probes; diagnostic, not information proof."""
    rng=np.random.default_rng(seed+701);groups=[]
    for name in ('base_train','base_val','query'):
        pieces=[]
        for c in range(7):
            a=np.flatnonzero((role==name)&(y==c))
            if len(a):pieces.extend(rng.choice(a,min(max_per_class,len(a)),replace=False).tolist())
        groups.extend(pieces)
    ids=np.array(groups);labels=y[ids];roles=role[ids];tr=roles=='base_train';va=roles=='base_val';qu=(roles=='query')&(labels>=5)
    values={};shapes={};projections={}
    for start in range(0,len(ids),16):
        v=batch(images,ids[start:start+16],[0],scaler,device)
        stage={'input':v}
        for j in range(4):v=encoder.body[j*3:j*3+3](v);stage[f'conv{j+1}']=v
        pooled=torch.stack([a.mean((2,3)) for a in torch.tensor_split(v,8,dim=2)],dim=2)[:,:,:,None]
        stage['pooled']=pooled;stage['embedding']=F.normalize(encoder.project(pooled),dim=-1)[:,:,None,None]
        for name,tensor in stage.items():
            shapes[name]=list(tensor.shape[1:]);flat=F.adaptive_avg_pool2d(tensor,(8,4)).flatten(1)
            if name not in projections:
                generator=torch.Generator(device='cpu');generator.manual_seed(seed+903)
                projections[name]=(torch.randn(flat.shape[1],64,generator=generator)/np.sqrt(64)).to(device)
            values.setdefault(name,[]).append((flat@projections[name]).cpu().numpy())
    report=[]
    for name,pieces in values.items():
        x=np.concatenate(pieces).astype(np.float64);mu=x[tr].mean(0);sd=x[tr].std(0).clip(1e-4);x=(x-mu)/sd
        a=np.c_[x,np.ones(len(x))];ridge=np.eye(a.shape[1]);ridge[-1,-1]=0
        from train_laspi_components import SIGNATURES
        targets=np.c_[np.eye(5)[labels[tr]],SIGNATURES[labels[tr]]]
        weight=np.linalg.solve(a[tr].T@a[tr]+ridge,a[tr].T@targets);pred=a@weight
        recalls=[];component_scores=[]
        for j in range(4):
            positives=qu&(SIGNATURES[labels,j]>.5)
            recalls.append(float((pred[positives,5+j]>=.5).mean()) if positives.any() else None)
            actual=SIGNATURES[labels[qu],j]>.5;decision=pred[qu,5+j]>=.5
            tp=int((actual&decision).sum());fp=int((~actual&decision).sum());fn=int((actual&~decision).sum())
            component_scores.append({'f1':2*tp/max(2*tp+fp+fn,1),'false_positive_rate':float(decision[~actual].mean()) if (~actual).any() else None})
        per_class=[]
        for c in range(5):
            p=pred[va,:5].argmax(1);truth=labels[va];tp=((p==c)&(truth==c)).sum();den=(p==c).sum()+(truth==c).sum();per_class.append(float(2*tp/max(den,1)))
        report.append({'stage':name,'shape':shapes[name],'source_val_macro_f1':float(np.mean(per_class)),'target_novel_positive_component_recall':recalls,'target_novel_component_scores':component_scores,'probe_dimensions':64,'fit_scope':'base_train only; query labels used exclusively for post-hoc reporting'})
    dump(folder/'information_probe.json',{'stages':report,'limitations':'Matched linear probes after pooling/random projection; neither mutual information nor proof of causal separation. Target diagnostics must not select checkpoints or thresholds.'})
    return report
