"""Read-only model evaluation: distinguish transfer, competition and adaptation.

Target labels are used for reporting only, never for fitting or checkpoint choice.
The report uses the first 5-shot support, not an independent statistical study.
"""
import argparse,json,sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax,expit
from prepare_laspi import ROOT,dump
from run_laspi_gfsl import load_samples
from run_laspi_cross_702010 import split_cross,cross_supports
from temporal_information import reserve_calibration
from train_laspi_components import SIGNATURES
from run_dggn_laspi import prepare_waveforms
from run_lwc_stability import fixed_memory,record_prefix,extract,classify
from lwc_stability_model import MODES,DualLWC


def five_f1(labels,logits):
    pred=logits.argmax(1);cm=np.bincount(5*labels+pred,minlength=25).reshape(5,5)
    f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    return {'macro_f1':float(f1.mean()),'per_class_f1':f1.tolist(),'confusion':cm.tolist()}


def components(probability,labels):
    truth=SIGNATURES[labels]>.5;out=[]
    for j in range(4):
        positive=truth[:,j];p=probability[:,j]
        out.append({'positive_mean_probability':float(p[positive].mean()) if positive.any() else None,
                    'negative_mean_probability':float(p[~positive].mean()) if (~positive).any() else None,
                    'positive_recall_at_half':float((p[positive]>=.5).mean()) if positive.any() else None,
                    'predicted_positive_fraction':float((p>=.5).mean())})
    return out


def diagnose(root):
    cfg=SimpleNamespace(**json.loads((root/'config.json').read_text(encoding='utf-8')))
    cfg.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    rows,y,_=load_samples(ROOT/cfg.reference_cache,metadata_only=True);wave=prepare_waveforms(cfg,rows);output=[]
    for case in cfg.cases:
        for seed in cfg.seeds:
            folder=root/f'{case}_s{seed}';role,*_=split_cross(rows,case,seed);role,_=reserve_calibration(rows,y,role,seed)
            memory=fixed_memory(rows,y,role,seed,cfg.memory_per_base_class)
            support=cross_supports(rows,y,role,case,seed,5,1)[0];novel=support['indices'][0]+support['indices'][1]
            val=record_prefix(rows,role,'base_val',cfg.validation_windows_per_csv);query=np.flatnonzero(role=='query');known=query[y[query]<5]
            ids=np.unique(np.r_[memory,novel,val,query]);lookup={int(i):j for j,i in enumerate(ids)}
            local=lambda v:np.array([lookup[int(i)] for i in v]);new=[local(v) for v in support['indices']]
            for mode in MODES:
                source_mode='frozen_exchange' if mode=='trainable_exchange' else mode
                for phase,path in [('source',folder/f'source_{source_mode}'/'best.pt'),('adapted',folder/mode/'adapted_5shot.pt')]:
                    model=DualLWC(cfg,mode).to(cfg.device);model.load_state_dict(torch.load(path,weights_only=True,map_location=cfg.device))
                    # Extraction always runs under no_grad. Updating flags do not affect GN.
                    values=extract(model,wave,ids,cfg.device,cfg.batch_size)
                    five=classify(values,local(memory),local(known),y[memory],cfg)
                    seven=classify(values,local(memory),local(query),y[memory],cfg,new)
                    mask=y[query]<5;pred=seven.argmax(1)
                    item={'condition':case,'seed':seed,'mode':mode,'phase':phase,
                          'target_base_only':five_f1(y[known],five),
                          'base_misrouted_to_novel_fraction':float((pred[mask]>=5).mean()),
                          'novel_misrouted_to_base_fraction':float((pred[~mask]<5).mean()),
                          'source_val_components':components(expit(values['d_logits'][local(val)]),y[val]),
                          'target_components':components(expit(values['d_logits'][local(query)]),y[query])}
                    for key in ('d','s'):
                        if key in values:
                            branch={key:values[key]};logits=classify(branch,local(memory),local(known),y[memory],cfg)
                            item[f'target_{key}_base_only']=five_f1(y[known],logits)
                    output.append(item)
                    print(f'{case} {mode}/{phase}: 仅五基类F1={item["target_base_only"]["macro_f1"]:.3f}, 基类误入新类={item["base_misrouted_to_novel_fraction"]:.1%}',flush=True)
                    del model
    dump(root/'diagnostics_5shot_first_support.json',output)


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);diagnose(parser.parse_args().result)
