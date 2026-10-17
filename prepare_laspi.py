"""Audit LASPI and build label-free STFT caches; no fitted normalization here."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parent
DEFAULT_SOURCE=Path(r'H:\keyan\9_23\LASPI-Detection_and_diagnostics_of_bearing_gear_and_combined_faults_of_gearbox\LASPI-Detection_and_diagnostics_of_bearing_gear_and_combined_faults_of_gearbox')
CLASSES=('Healthy_motor','Gear_surface_damage','Gear_half_broken_tooth',
    'Bearing_inner_race_fault','Bearing_outer_race_fault',
    'Gear_surface_and_bearing_inner_race_faults','Gear_half_broken_tooth_and_bearing_outer_race_faults')
CN=('健康','齿面损伤','半齿断裂','轴承内圈','轴承外圈','齿面＋内圈','半齿断裂＋外圈')
CHANNELS=('Ia_raw','Ib_raw','Ic_raw','acc_raw','Va_raw','Vb_raw','Vc_raw')
FS=25600; ANALYSIS_FS=12800; LENGTH=4096; NFFT=1024; HOP=256; WINDOWS=31


def dump(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.writing.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def csv_out(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def inventory(source):
    records=[]
    for label,name in enumerate(CLASSES):
        for p in sorted((source/name).rglob('*.csv')):
            match=re.fullmatch(r'(\d+)hz_(\d+)%_(\d+(?:\.\d+)?)rpm',p.parent.name,re.I)
            if not match:raise ValueError('Unverified condition filename: '+str(p))
            speed,load,rpm=match.groups()
            records.append({'record_id':f'{name}/{p.parent.name}/{p.stem}','condition_group':f'{name}/speed{speed}_load{load}',
                'label':label,'class_name':name,'speed_setting_hz':int(speed),'load_percent':int(load),
                'motor_rpm_filename':float(rpm),'file':str(p),'bytes':p.stat().st_size,
                'bearing_id':'unknown','acquisition_repeat_identity':'unknown'})
    if len(records)!=336:raise ValueError(f'Expected 336 CSV files, got {len(records)}')
    for label in range(7):
        for speed in (25,35,45):
            for load in (0,25,50,75):
                if sum(r['label']==label and r['speed_setting_hz']==speed and r['load_percent']==load for r in records)!=4:
                    raise ValueError('Unbalanced or missing original state/condition files')
    return records


def normalize_windows(chunks,mode='none'):
    centered=chunks-chunks.mean(-1,keepdims=True)
    if mode=='window_zscore':
        centered=centered/np.maximum(np.sqrt(np.mean(centered**2,axis=-1,keepdims=True)),1e-8)
    elif mode!='none':raise ValueError('Unknown normalization: '+mode)
    return centered


def prepare(args):
    torch.set_num_threads(2);start=time.perf_counter();destination=Path(args.output).resolve()
    normalization=getattr(args,'normalization','none')
    marker=destination/'complete.json'
    if marker.exists():
        previous=json.loads(marker.read_text(encoding='utf-8'))
        if previous['source']!=str(Path(args.source).resolve()):raise ValueError('Existing cache belongs to another source')
        if previous.get('waveform_normalization','none')!=normalization:raise ValueError('Cache normalization differs')
        print('CACHE_READY',destination,flush=True);return destination
    if destination.exists():raise FileExistsError('Incomplete cache retained; choose another output: '+str(destination))
    destination.mkdir(parents=True);records=inventory(Path(args.source).resolve());samples=[]
    total=len(records)*WINDOWS;shape=(total,7,NFFT//2+1,(LENGTH-NFFT)//HOP+1)
    images=np.lib.format.open_memmap(destination/'stft.npy',mode='w+',dtype=np.float16,shape=shape)
    wave=np.lib.format.open_memmap(destination/'waveform_examples.npy',mode='w+',dtype=np.float32,shape=(len(records),7,LENGTH))
    window=torch.hann_window(NFFT);raw_audit=[]
    for number,record in enumerate(records):
        file=Path(record['file']);x=np.loadtxt(file,delimiter=',',dtype=np.float32)
        if x.shape!=(256000,7) or not np.isfinite(x).all():raise ValueError('Invalid shape/finite values: '+str(file)+' '+str(x.shape))
        # Sampling conversion before chunking prevents per-window filter boundaries.
        x=resample_poly(x,1,2,axis=0).astype(np.float32)
        chunks=x[:WINDOWS*LENGTH].reshape(WINDOWS,LENGTH,7).transpose(0,2,1).copy()
        wave[number]=chunks[0]
        # Per-window DC removal; no label-dependent band selection or amplitude scaling.
        # Local transform, no test-fitted dataset statistic. Z-score removes RMS information.
        centered=normalize_windows(chunks,normalization)
        spec=torch.stft(torch.from_numpy(centered.reshape(-1,LENGTH)),NFFT,hop_length=HOP,
            win_length=NFFT,window=window,center=False,normalized=True,return_complex=True)
        image=torch.log1p(spec.abs().square()).numpy().reshape(WINDOWS,7,*shape[-2:])
        images[number*WINDOWS:(number+1)*WINDOWS]=image.astype(np.float16)
        record['sha256']=sha256(file);record['raw_samples']=256000;record['windows']=WINDOWS
        raw_audit.append({'record_id':record['record_id'],'channel_mean':x.mean(0).tolist(),
                          'channel_std':x.std(0).tolist(),'sha256':record['sha256']})
        for j in range(WINDOWS):
            samples.append({**{k:record[k] for k in ('record_id','condition_group','label','class_name','speed_setting_hz','load_percent','motor_rpm_filename')},
                'sample_id':record['record_id']+f'_w{j:03d}','window_index':j,'cache_index':number*WINDOWS+j,
                'start_s':j*LENGTH/ANALYSIS_FS,'duration_s':LENGTH/ANALYSIS_FS})
        if number%24==0 or number==len(records)-1:print(f'PREPARE {number+1}/{len(records)}',flush=True)
    images.flush();wave.flush();csv_out(destination/'records.csv',records);csv_out(destination/'samples.csv',samples)
    dump(destination/'raw_signal_audit.json',raw_audit)
    config={'source':str(Path(args.source).resolve()),'records':len(records),'samples':total,'classes':CLASSES,'classes_cn':CN,
        'raw_fs':FS,'analysis_fs':ANALYSIS_FS,'anti_alias':'scipy resample_poly 1/2 before chunking','window_length':LENGTH,
        'window_seconds':LENGTH/ANALYSIS_FS,'windows_per_record':WINDOWS,'unused_tail_seconds':10-WINDOWS*LENGTH/ANALYSIS_FS,
        'n_fft':NFFT,'hop_length':HOP,'center':False,'channels':CHANNELS,'shape':shape,
        'cache_dtype':'float16; normalization fitted later on known training only','physical_scaling':'raw sensor outputs; gains not applied',
        'waveform_normalization':normalization,'zscore_floor':1e-8 if normalization=='window_zscore' else None,
        'bearing_id_available':False,'repeat_files_not_assumed_independent':True,'elapsed_seconds':time.perf_counter()-start,
        'source_script_sha256':sha256(__file__)}
    dump(marker,config);print('PREPARED',destination,flush=True);return destination


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--output',type=Path,default=ROOT/'prepared')
    ap.add_argument('--normalization',choices=('none','window_zscore'),default='none');prepare(ap.parse_args())
