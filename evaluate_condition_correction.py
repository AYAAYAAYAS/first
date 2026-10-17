"""Exploratory source-only operating-condition correction on frozen features.

Fit a shared within-class linear nuisance effect on SOURCE condition-group
means, then remove it using known speed/load settings. No target fault label,
electrical erosion equation, physical loss, or model retraining is used.
"""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from prepare_laspi import ROOT,dump,sha256
from bayesian_layers import uncertainty
from run_laspi_gfsl import (load_samples,masks,fixed_distances,append_novel,support_ids,save_predictions,
    fit_temperature,metrics,auc,threshold_confidence,selective_metrics,sequence_indices,sequential,risk_coverage)


def correct_features(z,rows,train,ridge=1.):
    groups={}
    for idx in np.flatnonzero(train):groups.setdefault(rows[idx]['condition_group'],[]).append(idx)
    ids=[np.array(groups[name]) for name in sorted(groups)]
    labels=np.array([rows[a[0]]['label'] for a in ids])
    u=np.array([[rows[a[0]]['speed_setting_hz'],rows[a[0]]['load_percent']] for a in ids],dtype=np.float64)
    mean=u.mean(0);std=u.std(0).clip(1e-6);u=(u-mean)/std
    device=z.device;features=torch.stack([z[:,a].mean(1) for a in ids],dim=1)
    design=torch.tensor(u,dtype=z.dtype,device=device);within=design.clone();residual=features.clone()
    for label in range(5):
        which=np.flatnonzero(labels==label)
        within[which]-=design[which].mean(0);residual[:,which]-=features[:,which].mean(1)[:,None]
    precision=within.T@within+ridge*torch.eye(2,device=device,dtype=z.dtype)
    coefficients=torch.linalg.solve(precision,torch.einsum('gf,mgd->mfd',within,residual))
    all_u=np.array([[r['speed_setting_hz'],r['load_percent']] for r in rows],dtype=np.float64)
    normalized=torch.tensor((all_u-mean)/std,dtype=z.dtype,device=device)
    corrected=z-torch.einsum('nf,mfd->mnd',normalized,coefficients)
    audit={'fit_indices':np.flatnonzero(train).tolist(),'fit_groups':sorted(groups),'group_count':len(groups),
        'context_mean':mean.tolist(),'context_std':std.tolist(),'ridge':ridge,
        'inputs':['nominal speed setting Hz','nominal load percent'],'target_fault_labels_used':False,
        'fit':'equal-weight source known condition-group means, within-class centering',
        'common_nuisance_transfer_to_composites':'assumed; class-specific nonlinear effects not captured'}
    return corrected,coefficients,audit


def evaluate(reference,output=None,ridge=1.):
    torch.set_num_threads(2);start=time.perf_counter();reference=Path(reference).resolve()
    if not (reference/'execution.json').exists():raise ValueError('Wait for baseline execution to complete')
    config=json.loads((reference/'config.json').read_text(encoding='utf-8'))
    rows,y,_=load_samples(Path(config['cache']));models=json.loads((reference/'models.json').read_text(encoding='utf-8'))
    selected_modalities=sorted({r['modality'] for r in models})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output=Path(output).resolve() if output else reference/'condition_correction';original=output;number=2
    while output.exists():output=original.with_name(original.name+f'_run{number}');number+=1
    output.mkdir(parents=True);(output/'source_snapshot').mkdir();shutil.copy2(__file__,output/'source_snapshot'/Path(__file__).name)
    audit_config={**config,'output':str(output),'baseline_reference':str(reference),
        'methods':['encoder_mahalanobis','conditioned_mahalanobis','bayesian_conditioned_mahalanobis'],
        'method_names':{'encoder_mahalanobis':'原始CNN表示＋马氏','conditioned_mahalanobis':'工况修正＋确定性马氏','bayesian_conditioned_mahalanobis':'工况修正＋贝叶斯马氏'},
        'modalities':selected_modalities,'physics_constraints':False,'operating_metadata_used_at_inference':True,
        'ridge':ridge,'new_network_training':False,'exploratory':True,
        'reason_added':'Early held-out speed results exposed domain shift; this is an exploratory comparison, not an untouched confirmatory test',
        'parameter_selection':'ridge fixed before evaluating correction; no target-label tuning',
        'metadata_used_by_methods':['conditioned_mahalanobis','bayesian_conditioned_mahalanobis'],
        'source_script_sha256':sha256(__file__),'baseline_config_sha256':sha256(reference/'config.json')}
    dump(output/'config.json',audit_config);evaluations=[];unknown=[];display={}
    prediction_root=output/'predictions';prediction_root.mkdir()
    for scenario in config['scenarios']:
        train,val,pool,target,role=masks(rows,scenario);vi=np.flatnonzero(val);ti=np.flatnonzero(target)
        groups=sequence_indices(rows,ti)
        for seed in config['seeds']:
            for modality in selected_modalities:
                base_path=reference/'models'/f'{scenario}_{modality}_s{seed}'
                for bayesian,head in ((False,'encoder'),(False,'deterministic'),(True,'bayesian')):
                    name='encoder_mahalanobis' if head=='encoder' else 'bayesian_conditioned_mahalanobis' if bayesian else 'conditioned_mahalanobis'
                    if head=='encoder':
                        z=torch.tensor(np.load(base_path/'encoder_features.npy')[None],device=device)
                        corrected=z;coefficient=z.new_zeros((1,2,z.shape[-1]))
                        fit_audit={'fit_indices':np.flatnonzero(train).tolist(),'conditioning_enabled':False,'extra_network_training':False}
                    else:
                        z=torch.tensor(np.load(base_path/head/'posterior_features.npy'),device=device)
                        corrected,coefficient,fit_audit=correct_features(z,rows,train,ridge)
                    dump(output/f'fit_{scenario}_{modality}_{head}_s{seed}.json',fit_audit)
                    np.save(output/f'coefficient_{scenario}_{modality}_{head}_s{seed}.npy',coefficient.cpu().numpy())
                    base,pooled=fixed_distances(corrected,y,train,True)
                    vl=-base[:,vi].cpu().numpy();tt=fit_temperature(vl,y[vi]);vp=softmax(vl/tt,axis=-1)
                    ql=-base[:,ti].cpu().numpy();qp=softmax(ql/tt,axis=-1)
                    v,_,_,vmi=uncertainty(vp);p,_,_,qmi=uncertainty(qp)
                    vr=base[:,vi].mean(0).min(-1).values.cpu().numpy();qr=base[:,ti].mean(0).min(-1).values.cpu().numpy()
                    radius0=max(float(np.quantile(vr,.95)),1e-6);limit0=max(float(np.quantile(vmi,.95)),1e-6)
                    score=qr/radius0;combined=np.maximum(score,qmi/limit0) if bayesian else score
                    is_novel=y[ti]>=5
                    unknown.append({'scenario':scenario,'seed':seed,'modality':modality,'method':name,'k':0,
                        'distance_auroc':auc(is_novel,score),'distance_mi_auroc':auc(is_novel,combined),
                        'unknown_false_acceptance':float((combined[is_novel]<=1).mean()),'known_coverage':float((combined[~is_novel]<=1).mean())})
                    blocks=[];uqs=[];temps=[];accepted=[];sequence_predictions=[]
                    for k in config['shots']:
                        for episode in range(config['episodes']):
                            support=support_ids(rows,y,pool,scenario,seed,k,episode)
                            distance=append_novel(base,corrected,pooled,support,True)
                            v_logits=-distance[:,vi].cpu().numpy();q_logits=-distance[:,ti].cpu().numpy()
                            temperature=fit_temperature(v_logits,y[vi]);vp=softmax(v_logits/temperature,axis=-1);qp=softmax(q_logits/temperature,axis=-1)
                            p,h,e,mi=uncertainty(qp);v,_,_,vmi=uncertainty(vp);pred=p.argmax(1)
                            vd=distance[:,vi].mean(0).cpu().numpy();qd=distance[:,ti].mean(0).cpu().numpy()
                            threshold=threshold_confidence(y[vi],v,config['target_error'])
                            radius=float(np.quantile(vd[np.arange(len(vi)),y[vi]],.95));limit=float(np.quantile(vmi,.95))
                            keep_conf=p.max(1)>=threshold
                            keep=keep_conf & (qd[np.arange(len(ti)),pred]<=radius) & ((mi<=limit) if bayesian else True)
                            seq=sequential(qp,qd,y[ti],groups,threshold,radius,limit,bayesian)
                            sequence_predictions.append(np.array(seq.pop('predictions'),dtype=np.int8));seq.pop('true')
                            result={'scenario':scenario,'seed':seed,'modality':modality,'method':name,'k':k,'episode':episode,
                                'temperature':temperature,'confidence_threshold':threshold,'radius':radius,'mi_limit':limit,
                                'raw':metrics(y[ti],softmax(q_logits,axis=-1).mean(0)),'calibrated':metrics(y[ti],p),
                                'selective_confidence':selective_metrics(pred==y[ti],keep_conf),'selective_full':selective_metrics(pred==y[ti],keep),
                                'sequential':seq,'risk_coverage':risk_coverage(y[ti],p),'mean_mi':float(mi.mean())}
                            evaluations.append(result);blocks.append(p.astype(np.float64));uqs.append(np.column_stack([h,e,mi]).astype(np.float32))
                            temps.append(temperature);accepted.append(np.column_stack([keep_conf,keep]))
                            if scenario==config['scenarios'][0] and seed==config['seeds'][0] and k==5 and episode==0:
                                display[f'{modality}_{name}']={**result,'query_indices':ti.tolist(),'probabilities':p.tolist(),'mi':mi.tolist(),
                                    'entropy':h.tolist(),'accepted':keep.tolist()}
                    save_predictions(prediction_root/f'{scenario}_{modality}_{name}_s{seed}.npz',probabilities=np.stack(blocks),
                        uncertainties=np.stack(uqs),query_indices=ti,true_labels=y[ti],temperatures=np.array(temps),acceptance=np.stack(accepted),
                        shots=np.repeat(config['shots'],config['episodes']),episodes=np.tile(np.arange(config['episodes']),len(config['shots'])),
                        sequence_predictions=np.stack(sequence_predictions),sequence_true=np.array([y[ti][ids[0]] for _,ids in groups]))
                    print(f'CONDITION_CORRECTED {scenario} {modality} {head} seed={seed}',flush=True)
                    del z,corrected,base,pooled
                    if device.type=='cuda':torch.cuda.empty_cache()
        dump(output/'metrics.json',evaluations);dump(output/'unknown.json',unknown)
    dump(output/'metrics.json',evaluations);dump(output/'unknown.json',unknown);dump(output/'display.json',display)
    dump(output/'execution.json',{'completed':True,'evaluations':len(evaluations),'elapsed_seconds':time.perf_counter()-start,
                                 'encoders_retrained':0,'heads_retrained':0,'same_support_ids':True})
    print('FINISHED_CONDITION_CORRECTION',output,flush=True);return output


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('reference',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument('--ridge',type=float,default=1.);args=ap.parse_args();evaluate(args.reference,args.output,args.ridge)
