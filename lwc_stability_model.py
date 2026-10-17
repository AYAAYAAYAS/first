"""LWC discriminative/stable branches: four paired ablations, no BNN/physics law.

Wavelet follows the existing one/lwc_alpha_model.py LoG kernel formula.
The response image is scale-time, NOT a calibrated STFT spectrogram.
"""
import copy,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mahalanobis_metric import fit_reference,mahalanobis_logits,whiten

MODES=('single','parallel','frozen_exchange','trainable_exchange')

class LearnableLoG(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.size=cfg.wavelet_kernel;self.pool=cfg.feature_pool
        self.log_scale=nn.Parameter(torch.linspace(math.log(.8),math.log(100.),cfg.wavelets))
        self.shift=nn.Parameter(torch.zeros(cfg.wavelets));self.log_amplitude=nn.Parameter(torch.zeros(cfg.wavelets))
        self.register_buffer('time',torch.arange(self.size,dtype=torch.float32)-(self.size-1)/2)
    def kernels(self):
        scale=self.log_scale.exp().clamp(.6,self.size/3)
        shift=self.shift.tanh()*(self.size/8)
        u=(self.time[None]-shift[:,None])/scale[:,None]
        wave=(1-u.square())*torch.exp(-.5*u.square());wave=wave-wave.mean(-1,keepdim=True)
        return self.log_amplitude.exp().clamp(.25,4.)[:,None,None]*F.normalize(wave,dim=-1)[:,None]
    def forward(self,x):
        # Raw sampling stays 25.6kHz. Local response aggregation is a feature operation.
        response=F.conv1d(x[:,None],self.kernels(),padding=self.size//2)
        signed=F.avg_pool1d(response,self.pool,self.pool)
        energy=F.avg_pool1d(response.square(),self.pool,self.pool)
        image=torch.log1p(energy)
        sequence=torch.cat([signed,torch.sqrt(energy+1e-8)],dim=1)
        return sequence,image

class LWCEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.wavelet=LearnableLoG(cfg);self.tokens=cfg.tokens
        self.temporal=nn.Sequential(nn.Conv1d(2*cfg.wavelets,32,7,stride=2,padding=3,bias=False),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv1d(32,64,5,stride=2,padding=2,bias=False),nn.GroupNorm(8,64),nn.SiLU())
        self.image=nn.Sequential(nn.Conv2d(1,8,(3,5),stride=(1,2),padding=(1,2),bias=False),nn.GroupNorm(4,8),nn.SiLU(),nn.Conv2d(8,16,3,stride=(2,2),padding=1,bias=False),nn.GroupNorm(4,16),nn.SiLU())
        self.token_projection=nn.Sequential(nn.Linear(80,64),nn.LayerNorm(64),nn.SiLU())
        self.components=nn.ModuleList([nn.Linear(64,16) for _ in range(4)])
        self.heads=nn.ModuleList([nn.Linear(16,1) for _ in range(4)])
    def encode_tokens(self,x):
        sequence,image=self.wavelet(x)
        # Fixed disjoint bins avoid nondeterministic adaptive-pool CUDA backward.
        def pool(v):
            if v.shape[-1]<self.tokens:raise ValueError('token数超过时间位置数')
            return torch.stack([part.mean(-1) for part in torch.tensor_split(v,self.tokens,dim=-1)],dim=-1)
        one=pool(self.temporal(sequence))
        two=pool(self.image(image[:,None]).mean(2))
        return self.token_projection(torch.cat([one,two],dim=1).transpose(1,2))
    def summarize(self,tokens):
        global_feature=tokens.mean(1)
        blocks=torch.stack([torch.tanh(layer(global_feature)) for layer in self.components],dim=1)
        logits=torch.cat([head(blocks[:,j]) for j,head in enumerate(self.heads)],dim=1)
        return F.normalize(blocks.flatten(1),dim=-1),logits
    def forward(self,x):
        t=self.encode_tokens(x);z,l=self.summarize(t);return t,z,l

class DualLWC(nn.Module):
    def __init__(self,cfg,mode='parallel'):
        super().__init__();self.mode=mode;self.discriminative=LWCEncoder(cfg);self.stable=copy.deepcopy(self.discriminative)
        self.query_norm=nn.LayerNorm(64);self.reference_norm=nn.LayerNorm(64)
        self.attention=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.gate=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Linear(32,1));self.gate_max=cfg.gate_max
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2.)
    def forward(self,x,include_stable=None):
        td,zd,ld=self.discriminative(x)
        include_stable=(self.mode!='single') if include_stable is None else include_stable
        ts=zs=ls=None;gate=td.new_zeros(())
        if include_stable:
            if any(p.requires_grad for p in self.stable.parameters()):ts,zs,ls=self.stable(x)
            else:
                with torch.no_grad():ts,zs,ls=self.stable(x)
        if self.mode in ('frozen_exchange','trainable_exchange'):
            if ts is None:raise ValueError('交流模式缺少稳定分支')
            reference=ts if self.mode=='trainable_exchange' else ts.detach()
            context,_=self.attention(self.query_norm(td),self.reference_norm(reference),self.reference_norm(reference),need_weights=False)
            gate=self.gate_max*torch.sigmoid(self.gate(torch.cat([td,reference],dim=-1)))
            zd,ld=self.discriminative.summarize(td+gate*context)
        return {'d':zd,'d_logits':ld,'s':zs,'s_logits':ls,'gate':gate.mean()}

def configure_training(model,cfg,phase):
    for p in model.parameters():p.requires_grad_(False)
    # All variants use the same D update policy; frozen reference has its own LWC.
    for p in model.discriminative.parameters():p.requires_grad_(True)
    if phase=='support':
        for module in (model.discriminative.wavelet,model.discriminative.temporal[:3],model.discriminative.image[:3]):
            for p in module.parameters():p.requires_grad_(False)
    if phase=='pretrain' or (phase=='support' and model.mode=='trainable_exchange'):
        for p in model.stable.parameters():p.requires_grad_(True)
    if phase!='pretrain' and model.mode in ('frozen_exchange','trainable_exchange'):
        for module in (model.query_norm,model.reference_norm,model.attention,model.gate):
            for p in module.parameters():p.requires_grad_(True)
    groups=[]
    for branch in ('discriminative','stable'):
        encoder=getattr(model,branch);wave=[p for p in encoder.wavelet.parameters() if p.requires_grad];other=[p for name,p in encoder.named_parameters() if not name.startswith('wavelet.') and p.requires_grad]
        rate=cfg.source_lr if phase!='support' else (cfg.stable_adapt_lr if branch=='stable' else cfg.adapt_lr)
        if other:groups.append({'params':other,'lr':rate})
        if wave:groups.append({'params':wave,'lr':cfg.wavelet_lr if phase!='support' else cfg.stable_wavelet_adapt_lr})
    exchange=[p for name,p in model.named_parameters() if not name.startswith(('discriminative.','stable.')) and p.requires_grad]
    if exchange:groups.append({'params':exchange,'lr':cfg.source_lr if phase!='support' else cfg.adapt_lr})
    return torch.optim.Adam(groups,weight_decay=cfg.weight_decay)

def condition_alignment(z,labels,speeds):
    terms=[]
    for c in labels.unique():
        mask=labels==c;domains=speeds[mask].unique()
        if len(domains)<2:continue
        centers=torch.stack([z[mask&(speeds==d)].mean(0) for d in domains])
        terms.append((centers-centers.mean(0)).square().sum(-1).mean())
    return torch.stack(terms).mean() if terms else z.sum()*0

def episode_logits(z,labels,support,query,cfg):
    means,chol=fit_reference(z[support],labels[support],classes=5,shrinkage=cfg.shrinkage)
    return mahalanobis_logits(z[query],means,chol)/cfg.metric_temperature

def source_objective(model,x,labels,signatures,speeds,support,query,cfg,pretrain=False):
    out=model(x,include_stable=True if pretrain else None)
    dscore=episode_logits(out['d'],labels,support,query,cfg)
    bce=F.binary_cross_entropy_with_logits(out['d_logits'],signatures)
    alignment=x.new_zeros(())
    if pretrain:
        sscore=episode_logits(out['s'],labels,support,query,cfg)
        alignment=condition_alignment(out['s'],labels,speeds)
        loss=bce+cfg.metric_weight*F.cross_entropy(dscore,labels[query])
        loss+=F.binary_cross_entropy_with_logits(out['s_logits'],signatures)+cfg.metric_weight*F.cross_entropy(sscore,labels[query])+cfg.stability_weight*alignment
    else:
        score=dscore
        if model.mode!='single':score=.5*(score+episode_logits(out['s'],labels,support,query,cfg))
        loss=bce+cfg.metric_weight*F.cross_entropy(score,labels[query])
    return loss,{'bce':float(bce.detach()),'alignment':float(alignment.detach()),'gate':float(out['gate'].detach())}

def support_distance_logits(z,reference_z,labels,cfg):
    """Replay covariance only; LOO prototypes; alternate view for K=1."""
    base=labels<5;_,chol=fit_reference(z[base].detach(),labels[base],classes=5,shrinkage=cfg.shrinkage)
    counts=torch.bincount(labels,minlength=7)
    if (counts==0).any():raise ValueError('支持度量缺少类别')
    sums=torch.zeros(7,z.shape[1],device=z.device).index_add_(0,labels,reference_z)
    own=F.one_hot(labels,7).bool()&(counts[None]>1)
    centers=(sums[None]-own[:,:,None]*reference_z[:,None])/(counts[None]-own.to(counts.dtype)).clamp_min(1)[:,:,None]
    query=whiten(z,chol);prototypes=whiten(centers.reshape(-1,z.shape[1]),chol).reshape_as(centers)
    return -(query[:,None]-prototypes).square().sum(-1)/z.shape[1]/cfg.metric_temperature

def support_objective(model,teacher,x,labels,signatures,speeds,cfg,noise):
    out=model(x)
    with torch.no_grad():reference=model(x+noise)
    # Do not chase the reference branch through its prototype computation.
    score=support_distance_logits(out['d'],reference['d'],labels,cfg)
    base=labels<5;novel=~base
    loss=.5*F.binary_cross_entropy_with_logits(out['d_logits'][base],signatures[base])+.5*F.binary_cross_entropy_with_logits(out['d_logits'][novel],signatures[novel])
    with torch.no_grad():old=teacher(x[base])
    kd=(1-F.cosine_similarity(out['d'][base],old['d'],dim=-1)).mean()
    loss+=cfg.replay_distill_weight*kd
    alignment=x.new_zeros(());stable_bce=x.new_zeros(())
    if model.mode!='single':
        score=.5*(score+support_distance_logits(out['s'],reference['s'],labels,cfg))
        stable_bce=.5*F.binary_cross_entropy_with_logits(out['s_logits'][base],signatures[base])+.5*F.binary_cross_entropy_with_logits(out['s_logits'][novel],signatures[novel])
        alignment=condition_alignment(out['s'][base],labels[base],speeds[base])
        stable_kd=(1-F.cosine_similarity(out['s'][base],old['s'],dim=-1)).mean()
        loss+=cfg.stable_bce_weight*stable_bce+cfg.stability_weight*alignment+cfg.replay_distill_weight*stable_kd
    ce=F.cross_entropy(score,labels);loss+=cfg.metric_weight*ce
    if not torch.isfinite(loss):raise FloatingPointError('LWC双分支损失非有限')
    return loss,{'metric_ce':float(ce.detach()),'d_replay_drift':float(kd.detach()),'stable_bce':float(stable_bce.detach()),'alignment':float(alignment.detach()),'gate':float(out['gate'].detach())}
