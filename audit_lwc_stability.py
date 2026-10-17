"""Audit completed four-way experiments without training or writing figures."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from prepare_laspi import ROOT
from run_laspi_gfsl import load_samples
from run_laspi_cross_702010 import split_cross,cross_supports
from temporal_information import reserve_calibration
from run_lwc_stability import fixed_memory
from lwc_stability_model import MODES


def audit(root):
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    rows,y,_=load_samples(ROOT/cfg['reference_cache'],metadata_only=True)
    all_tasks=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    expected=len(cfg['cases'])*len(cfg['seeds'])*len(cfg['shots'])*cfg['support_repeats']*4
    assert len(all_tasks)==expected,(len(all_tasks),expected)
    assert json.loads((root/'execution.json').read_text(encoding='utf-8'))['finished']
    checked=0
    for case in cfg['cases']:
        for seed in cfg['seeds']:
            role,*_=split_cross(rows,case,seed);role,_=reserve_calibration(rows,y,role,seed)
            memory=fixed_memory(rows,y,role,seed,cfg['memory_per_base_class'])
            folder=root/f'{case}_s{seed}'
            records={}
            for i,r in enumerate(rows):records.setdefault(r['record_id'],set()).add(str(role[i]))
            assert all(len(v)==1 for v in records.values()),'同CSV窗口跨用途'
            reference_records={rows[i]['record_id'] for i in memory}
            query_records={rows[i]['record_id'] for i in np.flatnonzero(role=='query')}
            assert not reference_records&query_records
            assert len(memory)==5*cfg['memory_per_base_class']
            for k in cfg['shots']:
                for support in cross_supports(rows,y,role,case,seed,k,cfg['support_repeats']):
                    for c,group in zip((5,6),support['indices']):
                        assert len({rows[i]['record_id'] for i in group})==k
                        assert all(role[i]=='novel_support_pool' and y[i]==c for i in group)
                        assert not {rows[i]['record_id'] for i in group}&query_records
                    for mode in MODES:
                        task=json.loads((folder/'tasks'/f'{mode}_k{k}_e{support["episode"]}.json').read_text(encoding='utf-8'))
                        assert task['finished'] and task['source_unchanged']
                        assert task['support_sample_ids']==support['sample_ids'],'四组支持不配对'
                        update=task['update_audit'];assert update['discriminative_wavelet_max_change']==0
                        assert (update['stable_max_change']>0)==(mode=='trainable_exchange')
                        for name in ('raw','calibrated'):
                            report=task['report'][name]
                            assert all(np.isfinite(report[key]) for key in ('accuracy','macro_f1','base_f1','novel_f1','ece','nll'))
                            assert int(np.asarray(report['confusion']).sum())==int((role=='query').sum())
                        checked+=1
                # Directly inspect the saved example tensors, beyond scalar audit logs.
                for mode in MODES:
                    source_mode='frozen_exchange' if mode=='trainable_exchange' else mode
                    source=torch.load(folder/f'source_{source_mode}'/'best.pt',weights_only=True,map_location='cpu')
                    adapted=torch.load(folder/mode/f'adapted_{k}shot.pt',weights_only=True,map_location='cpu')
                    same=all(torch.equal(v,adapted[n]) for n,v in source.items() if n.startswith('stable.'))
                    assert same==(mode!='trainable_exchange')
                    assert all(torch.equal(v,adapted[n]) for n,v in source.items() if n.startswith('discriminative.wavelet.'))
    print(f'审计通过：{checked}/{expected}任务；CSV隔离、相同支持、有限指标、冻结/更新均一致。')


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True)
    audit(parser.parse_args().result)
