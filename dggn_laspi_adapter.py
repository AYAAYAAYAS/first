"""Auditable port of incomplete DGGN public code, NOT a verbatim paper reproduction.

Original ResidualBlock/MultiOrderDWConv and MSCAttention are executed unchanged.
Training bookkeeping/optimization is repaired here; upstream repository stays clean.
"""
import ast,copy,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def original_modules(repository):
    path=Path(repository)/'nets'/'feature_net_mff.py'
    spec=importlib.util.spec_from_file_location('dggn_original_mff',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Loading only this reviewed class avoids unrelated missing imports in model.py.
    source=Path(repository)/'model.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    definition=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MSCAttention')
    namespace={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),namespace)
    return module.ResidualBlock,namespace['MSCAttention']

class FixedLogFFT(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('mean',torch.zeros(1024));self.register_buffer('std',torch.ones(1024))
    @staticmethod
    def transform(x):
        power=torch.fft.rfft(x,norm='ortho').abs().square()[:,1:]
        if power.shape[1]!=4096:raise ValueError('固定对数频谱要求8192点波形')
        return torch.log1p(power.reshape(len(x),1024,4).mean(-1))
    def forward(self,x):return (self.transform(x)-self.mean)/self.std

class Backbone(nn.Module):
    def __init__(self,block,input_dim,encoding='raw'):
        super().__init__();self.frontend=FixedLogFFT() if encoding=='logfft' else nn.Identity()
        if encoding=='logfft':input_dim=1024
        self.encoder=nn.Sequential(block(input_dim,64),nn.ReLU(),block(64,128),nn.ReLU(),block(128,256),nn.ReLU(),block(256,512),nn.ReLU());self.out_dim=512
    def forward(self,x):return self.encoder(self.frontend(x))

class AdaptedDGGN(nn.Module):
    def __init__(self,repository,input_dim,device,encoding='raw'):
        super().__init__();block,attention=original_modules(repository)
        if encoding not in ('raw','logfft'):raise ValueError('input_encoding只能为raw/logfft')
        self.ca=Backbone(block,input_dim,encoding);self.cs=Backbone(block,input_dim,encoding);self.old_cs=None
        self.head=nn.Linear(512,5);self.fusion_head=nn.Linear(512,5)
        self.projector=nn.Sequential(nn.Linear(512,2048),nn.ReLU(),nn.Linear(2048,1024))
        self.predictor=nn.Linear(512,512);self.attention=attention(512,8,device)
        self.aux=None;self.stage=0;self.to(device)
    def incremental_copy(self):
        student=copy.deepcopy(self);student.stage=1
        student.old_cs=copy.deepcopy(self.cs).eval()
        for p in student.old_cs.parameters():p.requires_grad_(False)
        old=student.head;student.head=nn.Linear(1024,7,device=old.weight.device)
        with torch.no_grad():student.head.weight[:5,:512].copy_(old.weight);student.head.weight[:5,512:].zero_();student.head.bias[:5].copy_(old.bias)
        old=student.fusion_head;student.fusion_head=nn.Linear(512,7,device=old.weight.device)
        with torch.no_grad():student.fusion_head.weight[:5].copy_(old.weight);student.fusion_head.bias[:5].copy_(old.bias)
        student.aux=nn.Linear(512,3,device=old.weight.device)
        return student
    def train(self,mode=True):
        super().train(mode)
        if self.old_cs is not None:self.old_cs.eval()
        return self
    def outputs(self,x):
        ca=F.normalize(self.ca(x),dim=-1);cs=F.normalize(self.cs(x),dim=-1)
        features=cs if self.old_cs is None else torch.cat([F.normalize(self.old_cs(x),dim=-1),cs],dim=-1)
        fused=self.attention(ca.detach()[:,None],cs[:,None]).squeeze(1)
        return ca,cs,self.head(features),self.fusion_head(fused)

def contrastive(features,labels=None,temperature=.2):
    """Stable two-view SupCon (labels) or instance InfoNCE (no labels)."""
    n=features.shape[0]//2
    if len(features)!=2*n or n<2:raise ValueError('对比批必须包含至少两个样本的双视图')
    z=F.normalize(features,dim=-1);similarity=z@z.T/temperature
    eye=torch.eye(2*n,dtype=torch.bool,device=z.device)
    if labels is None:
        target=torch.arange(2*n,device=z.device);target=(target+n)%(2*n)
        return F.cross_entropy(similarity.masked_fill(eye,-1e9),target)
    repeated=labels.repeat(2);positive=(repeated[:,None]==repeated[None,:])&~eye
    log_probability=similarity-torch.logsumexp(similarity.masked_fill(eye,-1e9),dim=1,keepdim=True)
    return -((log_probability*positive).sum(1)/positive.sum(1).clamp_min(1)).mean()

def relational_distillation(student,teacher,temperature=1.):
    if len(student)<2:return student.sum()*0
    mask=torch.eye(len(student),dtype=torch.bool,device=student.device)
    a=(F.normalize(student,dim=-1)@F.normalize(student,dim=-1).T/temperature).masked_fill(mask,-1e9)
    b=(F.normalize(teacher,dim=-1)@F.normalize(teacher,dim=-1).T/temperature).masked_fill(mask,-1e9)
    return F.kl_div(a.log_softmax(-1),b.softmax(-1),reduction='batchmean')*temperature**2

def augment_waveforms(x,generator,noise_std=.01):
    # Public feature-vector reorder augmentation is not meaningful on raw samples:
    # circular shifts preserve cyclic waveform structure; small noise is a declared change.
    shifts=torch.randint(x.shape[1],(len(x),),generator=generator,device=x.device)
    positions=(torch.arange(x.shape[1],device=x.device)[None]-shifts[:,None])%x.shape[1]
    view=x.gather(1,positions)
    return view+noise_std*torch.randn(x.shape,generator=generator,device=x.device,dtype=x.dtype)

def train_loss(model,x,labels,cfg,generator,teacher=None):
    n=len(x);second=augment_waveforms(x,generator,cfg.augmentation_noise)
    both=torch.cat([x,second]);ca,cs,logits,fusion=model.outputs(both)
    targets=labels.repeat(2)
    supervised=contrastive(ca,labels,cfg.contrast_temperature)
    self_supervised=contrastive(model.projector(ca),None,cfg.contrast_temperature)
    classification=F.cross_entropy(logits,targets);fusion_loss=F.cross_entropy(fusion,targets)
    loss=supervised+cfg.ssl_weight*self_supervised+classification+cfg.fusion_weight*fusion_loss
    info={'supcon':float(supervised.detach()),'infonce':float(self_supervised.detach()),'class_ce':float(classification.detach()),'fusion_ce':float(fusion_loss.detach())}
    if teacher is not None:
        old=targets<5;new=~old
        with torch.no_grad():tca=F.normalize(teacher.ca(both[old]),dim=-1);tcs=F.normalize(teacher.cs(both[old]),dim=-1)
        relation=relational_distillation(ca[old],tca)+relational_distillation(cs[old],tcs)
        prediction=F.mse_loss(F.normalize(model.predictor(ca[old]),dim=-1),tca)
        aux_targets=torch.where(targets<5,torch.zeros_like(targets),targets-4)
        aux=F.cross_entropy(model.aux(cs),aux_targets)
        # Transfer operates only on the two novel classes, with correct old-class offset.
        transfer=F.kl_div((logits[new,5:]/2).log_softmax(-1),(fusion[new,5:].detach()/2).softmax(-1),reduction='batchmean')*4
        loss+=cfg.relation_weight*relation+cfg.prediction_weight*prediction+cfg.aux_weight*aux+cfg.transfer_weight*transfer
        info.update(relation=float(relation.detach()),prediction=float(prediction.detach()),aux=float(aux.detach()),transfer=float(transfer.detach()))
    if not torch.isfinite(loss):raise FloatingPointError('DGGN适配损失出现非有限值：'+str(info))
    return loss,info

@torch.no_grad()
def embeddings(model,waveforms,indices,device,batch_size=64,which='ca'):
    model.eval();out=[]
    for start in range(0,len(indices),batch_size):
        x=torch.from_numpy(waveforms[np.asarray(indices[start:start+batch_size])].astype(np.float32)).to(device)
        out.append(F.normalize(getattr(model,which)(x),dim=-1).cpu().numpy())
    return np.concatenate(out)

def boundary_exemplars(features,indices,count):
    """Public BAEP far-from-current-center-residual ordering, with aligned indices."""
    count=min(count,len(indices));features=features.astype(np.float64);features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    center=features.mean(0);center/=max(np.linalg.norm(center),1e-12);summed=np.zeros(features.shape[1]);selected=[]
    for k in range(1,count+1):
        scores=np.square(center-(features+summed)/k).sum(1);scores[selected]=-np.inf
        index=int(scores.argmax());selected.append(index);summed+=features[index]
    return np.asarray(indices)[selected]
