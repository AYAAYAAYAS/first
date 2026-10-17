"""振动跨工况GFSL：rpm近似阶次轴、混合shot、部分解冻、共享马氏距离。

运行参数集中在 arguments()；不读取电压/电流作为模型输入。
原始CSV 70/20/10划分复用已核实协议，新结果与旧实验分开。
"""
from __future__ import annotations
import argparse, collections, json, os, shutil, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax
from prepare_laspi import ROOT, ANALYSIS_FS, NFFT, dump, csv_out, sha256
from run_laspi_gfsl import (Encoder, load_samples, batch, encode, seed_all, save_checkpoint, fit_temperature, save_predictions)
from run_laspi_same_condition import save_feature_array, summarize
from run_laspi_cross_702010 import CASES, split_cross, evaluate_cross
from bayesian_layers import FeatureHead
from mahalanobis_metric import fit_reference, mahalanobis_logits


def vibration_images(raw, rows, axis, order_max):
    """仅频率轴重映射。无转速脉冲；按文件名rpm假设每条记录恒速。"""
    vibration = np.asarray(raw[:, 3:4]).copy()
    if axis == 'hz':
        return vibration, np.arange(vibration.shape[2])*ANALYSIS_FS/NFFT
    rotor_hz = np.array([r['motor_rpm_filename']/60 for r in rows])
    if not np.isfinite(rotor_hz).all() or (rotor_hz <= 0).any():
        raise ValueError('阶次轴需要有效的文件名rpm，不能用变频器设置值代替')
    if order_max*rotor_hz.max() > ANALYSIS_FS/2:
        raise ValueError('order-max超过最高转速下的奈奎斯特覆盖范围')
    orders = np.linspace(0, order_max, vibration.shape[2])
    # 先还原功率再插值，最后log1p，避免在对数幅值上插值。
    for hz in sorted(set(rotor_hz)):
        ids = np.flatnonzero(rotor_hz == hz)
        positions = orders*hz/(ANALYSIS_FS/NFFT)
        lower = np.floor(positions).astype(int)
        upper = np.minimum(lower+1, vibration.shape[2]-1)
        fraction = (positions-lower).astype(np.float32)[None, :, None]
        for begin in range(0, len(ids), 128):
            chosen = ids[begin:begin+128]
            power = np.expm1(vibration[chosen, 0].astype(np.float32))
            mapped = power[:, lower]*(1-fraction)+power[:, upper]*fraction
            vibration[chosen, 0] = np.log1p(mapped).astype(np.float16)
        print(f'阶次转换：轴转频率={hz:.3f} Hz，rpm={60*hz:.1f}，窗口={len(ids)}', flush=True)
    return vibration, orders


def scaler_for_vibration(images, train):
    ids = np.flatnonzero(train); total = 0.; squared = 0.; count = 0
    for begin in range(0, len(ids), 128):
        x = images[ids[begin:begin+128]].astype(np.float64)
        total += x.sum(); squared += (x*x).sum(); count += x.size
    mean = total/count; std = np.sqrt(max(squared/count-mean*mean, 1e-8))
    return {'mean':[mean], 'std':[std], 'fit_indices':ids.tolist(), 'fit_scope':'source base_train only'}


def mixed_episode(y, role, cond, conditions, rng, train_shots):
    k = int(rng.choice(train_shots)); ids = []; support = []; query = []
    for c in range(5):
        available = [name for name in conditions
                     if np.sum((role == 'base_train') & (cond == name) & (y == c)) >= max(k, 7)]
        first, second = rng.choice(available, 2, replace=False)
        s = np.flatnonzero((role == 'base_train') & (cond == first) & (y == c))
        q = np.flatnonzero((role == 'base_train') & (cond == second) & (y == c))
        start = len(ids)
        ids.extend(rng.choice(s, k, replace=False)); ids.extend(rng.choice(q, 7, replace=False))
        support.extend(range(start, start+k)); query.extend(range(start+k, start+k+7))
    return np.array(ids), np.array(support), np.array(query), k


def episode_reference(z, labels, support, bank, bank_labels, keep, shrinkage):
    # 协方差收缩系数在这里传入：同一个args.shrinkage用于训练/验证/测试。
    # 排除当前episode窗口的源训练参考池，避免1-shot零协方差；池每轮刷新。
    _, chol = fit_reference(bank[keep].detach(), bank_labels[keep], shrinkage=shrinkage)
    means = torch.stack([z[support][labels[support] == c].mean(0) for c in range(5)])
    return means, chol


def validate(draws, y, role, shrinkage):
    tr = np.flatnonzero(role == 'base_train'); va = np.flatnonzero(role == 'base_val')
    labels = torch.as_tensor(y, device=draws.device); probabilities = []
    for z in draws:
        means, chol = fit_reference(z[tr], labels[tr], shrinkage=shrinkage)
        probabilities.append(mahalanobis_logits(z[va], means, chol).softmax(-1))
    p = torch.stack(probabilities).mean(0); target = labels[va]
    nll = float(-p[torch.arange(len(va), device=p.device), target].clamp_min(1e-12).log().mean())
    accuracy = float((p.argmax(-1) == target).float().mean())
    return nll, accuracy


def state_cpu(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train_backbone(images, rows, y, role, cond, conditions, scaler, args, seed, folder, device):
    seed_all(seed); model = Encoder(1).to(device); model.classifier = nn.Identity()
    if (folder/'encoder_features.npy').exists():
        model.load_state_dict(torch.load(folder/'best_encoder.pt', map_location=device, weights_only=True))
        return model, np.load(folder/'encoder_features.npy')
    # 编码器预训练学习率：--encoder-lr；轮数：--encoder-epochs。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.encoder_lr)
    rng = np.random.default_rng(seed); tr = np.flatnonzero(role == 'base_train')
    bank_labels = torch.as_tensor(y[tr], device=device); history = []; best = float('inf')
    for epoch in range(1, args.encoder_epochs+1):
        bank = torch.as_tensor(encode(model, images, tr, [0], scaler, device), device=device)
        model.train(); losses = []; counts = collections.Counter()
        for _ in range(args.encoder_steps):
            ids, support, query, k = mixed_episode(y, role, cond, conditions, rng, args.train_shots)
            yy = torch.as_tensor(y[ids], device=device); keep = torch.as_tensor(~np.isin(tr, ids), device=device)
            optimizer.zero_grad(set_to_none=True)
            z = model(batch(images, ids, [0], scaler, device))
            means, chol = episode_reference(z, yy, support, bank, bank_labels, keep, args.shrinkage)
            loss = F.cross_entropy(mahalanobis_logits(z[query], means, chol), yy[query], label_smoothing=args.label_smoothing)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            losses.append(float(loss.detach())); counts[k] += 1
        selected = np.flatnonzero((role == 'base_train') | (role == 'base_val'))
        values = encode(model, images, selected, [0], scaler, device)
        with torch.no_grad():
            nll, acc = validate(torch.as_tensor(values, device=device)[None], y[selected], role[selected], args.shrinkage)
        history.append({'epoch':epoch, 'train_loss':float(np.mean(losses)), 'base_val_nll':nll, 'base_val_accuracy':acc, 'shot_counts':dict(counts)})
        if nll < best:
            best = nll; save_checkpoint(state_cpu(model), folder/'best_encoder.pt')
        dump(folder/'encoder_history.json', history)
        print(f'编码器 {folder.name} {epoch}/{args.encoder_epochs}：loss={np.mean(losses):.4f} 源验证Acc={acc:.4f} NLL={nll:.4f} 混合shot={dict(counts)}', flush=True)
    model.load_state_dict(torch.load(folder/'best_encoder.pt', weights_only=True, map_location=device))
    features = encode(model, images, np.arange(len(rows)), [0], scaler, device)
    save_feature_array(folder/'encoder_features.npy', features)
    return model, features


def train_joint(backbone, initial_features, images, rows, y, role, cond, conditions, scaler, args, seed, folder, bayesian, device):
    folder.mkdir(exist_ok=True)
    if (folder/'posterior_features.npy').exists():
        return np.load(folder/'posterior_features.npy')
    seed_all(seed); encoder = Encoder(1).to(device); encoder.classifier = nn.Identity()
    encoder.load_state_dict(backbone.state_dict())
    for p in encoder.parameters(): p.requires_grad_(False)
    if args.unfreeze == 'last':
        # 仅解冻第4卷积块（body[9:12]）及128维投影；前3块保持冻结。
        for module in (encoder.body[-3:], encoder.project):
            for p in module.parameters(): p.requires_grad_(True)
    head = FeatureHead(bayesian, args.prior_std).to(device)
    tr = np.flatnonzero(role == 'base_train'); selected = np.flatnonzero((role == 'base_train') | (role == 'base_val'))
    mu = initial_features[tr].mean(0); sd = initial_features[tr].std(0).clip(1e-3)
    # 特征标准化固定为预训练源基类统计，不使用验证/测试拟合。
    mean = torch.as_tensor(mu, device=device); std = torch.as_tensor(sd, device=device)
    groups = [{'params':head.parameters(), 'lr':args.head_lr}]
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    if trainable: groups.append({'params':trainable, 'lr':args.finetune_lr})
    optimizer = torch.optim.Adam(groups); rng = np.random.default_rng(seed)
    print(f'联合训练 {folder.name}：头lr={args.head_lr}，编码器lr={args.finetune_lr}，解冻参数={sum(p.numel() for p in trainable)}，轮数={args.head_epochs}', flush=True)
    bank_labels = torch.as_tensor(y[tr], device=device); history = []; best = float('inf')
    for epoch in range(1, args.head_epochs+1):
        bank = torch.as_tensor(encode(encoder, images, tr, [0], scaler, device), device=device)
        bank = (bank-mean)/std; losses = []; counts = collections.Counter()
        encoder.train(); head.train()
        for _ in range(args.head_steps):
            ids, support, query, k = mixed_episode(y, role, cond, conditions, rng, args.train_shots)
            yy = torch.as_tensor(y[ids], device=device); keep = torch.as_tensor(~np.isin(tr, ids), device=device)
            optimizer.zero_grad(set_to_none=True)
            x = (encoder(batch(images, ids, [0], scaler, device))-mean)/std
            total = 0.; draws = args.mc_train if bayesian else 1
            for _ in range(draws):
                head.resample()
                with torch.no_grad(): reference_bank = head(bank)
                z = head(x)
                means, chol = episode_reference(z, yy, support, reference_bank, bank_labels, keep, args.shrinkage)
                total += F.cross_entropy(mahalanobis_logits(z[query], means, chol), yy[query], label_smoothing=args.label_smoothing)
            total = total/draws + args.kl_beta*head.kl()/head.count()
            total.backward(); nn.utils.clip_grad_norm_(list(head.parameters())+trainable, 5.); optimizer.step()
            losses.append(float(total.detach())); counts[k] += 1
        features = encode(encoder, images, selected, [0], scaler, device)
        with torch.no_grad():
            xval = (torch.as_tensor(features, device=device)-mean)/std; values = []
            for _ in range(args.mc_val if bayesian else 1):
                head.resample(); values.append(head(xval))
            nll, acc = validate(torch.stack(values), y[selected], role[selected], args.shrinkage)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'base_val_nll':nll,'base_val_accuracy':acc,'shot_counts':dict(counts)})
        if nll < best:
            best = nll; save_checkpoint({'encoder':state_cpu(encoder),'head':state_cpu(head)}, folder/'best_joint.pt')
        dump(folder/'head_history.json', history)
        print(f'联合 {folder.name} {epoch}/{args.head_epochs}：loss={np.mean(losses):.4f} 源验证Acc={acc:.4f} NLL={nll:.4f} 混合shot={dict(counts)}', flush=True)
    saved = torch.load(folder/'best_joint.pt', map_location=device, weights_only=True)
    encoder.load_state_dict(saved['encoder']); head.load_state_dict(saved['head'])
    features = encode(encoder, images, np.arange(len(rows)), [0], scaler, device)
    save_feature_array(folder/'joint_encoder_features.npy', features)
    dump(folder/'feature_scaler.json', {'mean':mu.tolist(),'std':sd.tolist(),'fit_indices':tr.tolist()})
    values = []; weights = []
    with torch.no_grad():
        x = (torch.as_tensor(features, device=device)-mean)/std
        for _ in range(args.mc_eval if bayesian else 1):
            head.resample(); values.append(head(x).cpu().numpy())
            weights.append({f'{i}_{k}':getattr(layer,k+'_sample').detach().cpu().clone() for i,layer in enumerate(head.layers) for k in ('weight','bias')})
    save_checkpoint(weights, folder/'posterior_weight_samples.pt')
    posterior = np.stack(values); save_feature_array(folder/'posterior_features.npy', posterior)
    return posterior


def report(result, records, diagnostics, args):
    summary = summarize(result, records)
    lines = ['# 振动跨工况GFSL：阶次表征与部分解冻', '',
             f'输入轴={args.axis}；仅振动；评价shot={args.shots}；训练混合shot={args.train_shots}。',
             f'收缩={args.shrinkage}；编码器lr={args.encoder_lr}；头lr={args.head_lr}；微调lr={args.finetune_lr}；解冻={args.unfreeze}。',
             f'编码器{args.encoder_epochs}轮×{args.encoder_steps}步；联合{args.head_epochs}轮×{args.head_steps}步。',
             'CSV数量228/65/33；主要分数仅未见工况子集。两复合类每类K个窗口来自K个不同源训练CSV，不等于K个独立轴承。',
             '阶次轴采用文件名电机rpm恒速近似；没有同步转速脉冲。阶次范围固定，时间轴仍为秒。',
             '协方差来自源基类池；训练池每轮刷新且排除当次episode窗口，池特征与协方差不回传梯度；支持只拟合原型。',
             '训练和模型选择只用源基类；校准只用源验证基类。BNN仅在特征头，不保证未知域的风险上界。', '',
             '|头|shot|Macro-F1|复合F1|ECE|确认覆盖|确认错误|', '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        error = '无确认' if s['selective_error'] is None else f"{s['selective_error']:.2%}"
        lines.append(f"|{s['method']}|{s['k']}|{s['macro_f1']:.2%}|{s['novel_f1']:.2%}|{s['ece']:.4f}|{s['coverage']:.2%}|{error}|")
    (result/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    dump(result/'known_only_metrics.json', diagnostics)
    return summary


def base_scores(y, p):
    """五类指标，不沿用GFSL指标中的复合类切片。"""
    pred = p.argmax(-1); cm = np.bincount(5*y+pred, minlength=25).reshape(5,5)
    f1 = 2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    confidence = p.max(-1); correct = pred == y; ece = 0.
    for j in range(10):
        mask = (confidence >= j/10) & ((confidence < (j+1)/10) if j < 9 else (confidence <= 1))
        if mask.any(): ece += mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return {'accuracy':float(correct.mean()),'macro_f1':float(f1.mean()),'ece':float(ece),
            'nll':float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean()),
            'confusion':cm.tolist(),'per_class_f1':f1.tolist(),
            'per_class_recall':(np.diag(cm)/np.maximum(cm.sum(1),1)).tolist()}


def evaluate_base(z, y, role, target, args, seed, method, result):
    tr = np.flatnonzero(role == 'base_train'); va = np.flatnonzero(role == 'base_val')
    query = np.flatnonzero((role == 'query') & (y < 5)); val_logits = []; test_logits = []
    with torch.no_grad():
        for draw in z:
            values = torch.as_tensor(draw)
            means, chol = fit_reference(values[tr],torch.as_tensor(y[tr]),shrinkage=args.shrinkage)
            val_logits.append(mahalanobis_logits(values[va],means,chol).numpy())
            test_logits.append(mahalanobis_logits(values[query],means,chol).numpy())
    temperature = fit_temperature(np.stack(val_logits),y[va])
    logits = np.stack(test_logits); raw = softmax(logits,axis=-1).mean(0)
    calibrated = softmax(logits/temperature,axis=-1).mean(0)
    save_predictions(result/'predictions'/f'base_vibration_{method}_{target}_s{seed}.npz',
                     probabilities=calibrated,raw_probabilities=raw,true_labels=y[query],query_indices=query)
    return {'condition':target,'seed':seed,'method':method,'temperature':temperature,
            'calibrated':base_scores(y[query],calibrated),'raw':base_scores(y[query],raw),
            'scope':'unseen target condition, five base candidates only; no composite supports'}


def report_base(result, records, args):
    summary = []
    for method in ('deterministic','bayesian'):
        selected = [r for r in records if r['method'] == method]
        if not selected: continue
        summary.append({'method':method,'task_seed_count':len(selected),
                        **{key:float(np.mean([r['calibrated'][key] for r in selected])) for key in ('accuracy','macro_f1','ece','nll')},
                        'raw_ece':float(np.mean([r['raw']['ece'] for r in selected]))})
    dump(result/'summary.json',summary)
    lines = ['# 仅五基类跨工况诊断', '',
             f'输入轴={args.axis}；编码器解冻={args.unfreeze}；收缩={args.shrinkage}。',
             '保留原326份CSV的70/20/10划分；本实验只使用其中五基类训练、验证和目标测试样本，复合类不构建支持原型、不参与候选竞争。',
             '编码器先预训练；unfreeze=none表示预训练完成后整个编码器冻结，仍训练确定性/贝叶斯头。',
             f'训练混合支持数={args.train_shots}；测试五基类使用全部源基类训练原型，不是各shot GFSL成绩。',
             '温度只在源基类验证集拟合；同时报告未校准ECE。阶次是文件名rpm恒速近似，Hz/阶次覆盖范围不同。', '',
             '|头|准确率|Macro-F1|未校准ECE|校准ECE|', '|---|---:|---:|---:|---:|']
    for s in summary: lines.append(f"|{s['method']}|{s['accuracy']:.2%}|{s['macro_f1']:.2%}|{s['raw_ece']:.4f}|{s['ece']:.4f}|")
    lines.extend(['','## 25 Hz空载外圈诊断','','|头|种子|外圈召回|外圈误判为健康|外圈误判为内圈|','|---|---:|---:|---:|---:|'])
    for record in records:
        if record['condition'] != '25hz_0pct': continue
        outer = np.asarray(record['calibrated']['confusion'])[4]; total = max(int(outer.sum()),1)
        lines.append(f"|{record['method']}|{record['seed']}|{outer[4]/total:.2%}|{outer[0]/total:.2%}|{outer[3]/total:.2%}|")
    (result/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run(args):
    torch.set_num_threads(2); started = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    default_name = f'vibration_{args.axis}_base_{args.unfreeze}' if args.base_only else 'vibration_order_mixed15'
    result = (args.output or ROOT/'results'/default_name).resolve()
    if result.exists() and not args.resume:
        original = result; number = 2
        while result.exists(): result = original.with_name(original.name+f'_run{number}'); number += 1
    for name in ('models','splits','predictions','source_snapshot'): (result/name).mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','no_show','output')}
    config.update({'output':str(result),'protocol':'cross_condition_702010','modalities':['vibration'],'normalizations':['window_zscore'],
                   'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                   'covariance_pool':'source base train, epoch refresh, excludes episode windows',
                   'order_reference':'motor_rpm_filename / 60; constant-speed approximation, no tachometer'})
    sources = ('run_laspi_vibration_order.py','show_laspi_vibration_order.py','run_laspi_cross_702010.py','run_laspi_same_condition.py','run_laspi_gfsl.py','prepare_laspi.py','bayesian_layers.py','mahalanobis_metric.py')
    config['code_hashes'] = {s:sha256(ROOT/s) for s in sources}
    if args.resume:
        old = json.loads((result/'config.json').read_text(encoding='utf-8'))
        if old != config: raise ValueError('续跑参数或代码发生变化，请使用新的输出目录')
    else:
        dump(result/'config.json', config)
        for source in sources: shutil.copy2(ROOT/source, result/'source_snapshot'/source)
    print(f'设备：{device} {config["gpu"] or ""}；结果目录：{result}', flush=True)
    print(f'仅振动，轴={args.axis}，收缩={args.shrinkage}，混合训练={args.train_shots}，评价={args.shots}', flush=True)
    if args.base_only: print('五基类模式：不构建复合支持原型，不评价GFSL shot；--train-shots仍用于源基类episode训练。',flush=True)
    rows, y, raw = load_samples(ROOT/'prepared_window_zscore', mmap=False)
    images, axis = vibration_images(raw, rows, args.axis, args.order_max)
    examples = [next(i for i,r in enumerate(rows) if r['label'] == 3 and r['speed_setting_hz'] == speed and r['load_percent'] == 0 and not r['excluded_duplicate']) for speed in (25,35,45)]
    with (result/'axis_examples.npz').open('wb') as stream:
        np.savez_compressed(stream, rpm=np.array([rows[i]['motor_rpm_filename'] for i in examples]),
                            hz_axis=np.arange(raw.shape[2])*ANALYSIS_FS/NFFT, hz_spectrum=raw[examples,3].mean(-1),
                            input_axis=axis, input_spectrum=images[examples,0].mean(-1))
    del raw
    save_feature_array(result/'input_axis.npy', axis)
    print(f'振动输入：{images.shape}，内存={images.nbytes/2**20:.1f} MiB；预训练后冻结方式={args.unfreeze}（none为整个编码器冻结）', flush=True)
    records = json.loads((result/'metrics.json').read_text(encoding='utf-8')) if args.resume and (result/'metrics.json').exists() else []
    diagnostics = json.loads((result/'known_only_metrics.json').read_text(encoding='utf-8')) if args.resume and (result/'known_only_metrics.json').exists() else []
    total = len(args.cases)*len(args.seeds)*2
    for target in args.cases:
        for seed in args.seeds:
            role, partition, cond, conditions, audit = split_cross(rows, target, seed)
            dump(result/'splits'/f'{target}_s{seed}.json', audit)
            csv_out(result/'splits'/f'{target}_s{seed}.csv', [{'sample_id':r['sample_id'],'record_id':r['record_id'],'label':int(y[i]),'partition':partition[i],'role':role[i]} for i,r in enumerate(rows)])
            # 运行前确认15-shot实际有足够不同源CSV，而非固定假设32个。
            for c in (() if args.base_only else (5,6)):
                count = len({rows[i]['record_id'] for i in np.flatnonzero((role == 'novel_support_pool') & (y == c))})
                if max(args.shots) > count: raise ValueError(f'类别{c}只有{count}个支持CSV，无法{max(args.shots)}-shot')
            print(f'工况={target} 种子={seed} CSV划分={audit["actual_record_counts"]}', flush=True)
            folder = result/'models'/f'vibration_{target}_s{seed}'; folder.mkdir(exist_ok=True)
            scaler = scaler_for_vibration(images, role == 'base_train'); dump(folder/'image_scaler.json', scaler)
            encoder, features = train_backbone(images, rows, y, role, cond, conditions, scaler, args, seed, folder, device)
            for method in ('deterministic','bayesian'):
                if any((r['condition'],r['seed'],r['method']) == (target,seed,method) for r in records): continue
                z = train_joint(encoder, features, images, rows, y, role, cond, conditions, scaler, args, seed, folder/method, method == 'bayesian', device)
                if args.base_only:
                    record = evaluate_base(z,y,role,target,args,seed,method,result)
                    records.append(record); dump(result/'metrics.json',records)
                    summary = report_base(result,records,args)
                    score = record['calibrated']
                    print(f'五基类测试 {target} {method}：Acc={score["accuracy"]:.4f} MacroF1={score["macro_f1"]:.4f} 原始ECE={record["raw"]["ece"]:.4f} 校准ECE={score["ece"]:.4f}',flush=True)
                    outer = np.asarray(score['confusion'])[4]; outer_total = max(int(outer.sum()),1)
                    print(f'外圈诊断 {target} {method}：召回={outer[4]/outer_total:.2%}；误判去向比例：健康={outer[0]/outer_total:.2%} 齿面={outer[1]/outer_total:.2%} 半齿断裂={outer[2]/outer_total:.2%} 内圈={outer[3]/outer_total:.2%}',flush=True)
                else:
                    new, diagnostic = evaluate_cross(z, rows, y, role, target, args, seed, 'vibration', method, result)
                    records.extend(new); diagnostics.append(diagnostic); dump(result/'metrics.json', records)
                    summary = report(result, records, diagnostics, args)
                    for k in args.shots:
                        subset = [r for r in new if r['k'] == k]
                        print(f'测试 {target} {method} {k}-shot：MacroF1={np.mean([r["calibrated"]["macro_f1"] for r in subset]):.4f} 复合F1={np.mean([r["calibrated"]["novel_f1"] for r in subset]):.4f} ECE={np.mean([r["calibrated"]["ece"] for r in subset]):.4f}', flush=True)
                completed = len({(r['condition'],r['seed'],r['method']) for r in records})
                dump(result/'progress.json', {'completed_groups':completed,'total_groups':total,'finished':False})
                print(f'完成 {completed}/{total} 组', flush=True)
    dump(result/'execution.json', {'finished':True,'elapsed_seconds':time.perf_counter()-started,'gpu':config['gpu'],'evaluations':len(records)})
    dump(result/'progress.json', {'completed_groups':total,'total_groups':total,'finished':True})
    print(f'全部完成：{result}', flush=True)
    if not args.no_show:
        from show_laspi_vibration_order import show
        show(result)
    return result


def arguments(argv=None):
    # ===== 可调参数集中在这里；命令行同名参数优先于默认值 =====
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path); ap.add_argument('--resume', action='store_true'); ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--base-only', action='store_true', help='仅五基类诊断，复合类不建原型且不参与测试候选')
    ap.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES)); ap.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    ap.add_argument('--shots', nargs='+', type=int, default=[1,5,10,15,20])
    ap.add_argument('--train-shots', nargs='+', type=int, default=[1,5,10,15,20])
    ap.add_argument('--axis', choices=('order','hz'), default='order'); ap.add_argument('--order-max', type=float, default=128.)
    ap.add_argument('--unfreeze', choices=('last','none'), default='last')
    ap.add_argument('--shrinkage', type=float, default=0.3, help='共享协方差收缩系数：(1-lambda)C+lambda*平均方差*I')
    ap.add_argument('--encoder-lr', type=float, default=0.0003, help='编码器预训练学习率')
    ap.add_argument('--head-lr', type=float, default=0.001, help='确定性/BNN头学习率')
    ap.add_argument('--finetune-lr', type=float, default=0.0001, help='解冻编码器部分的联合训练学习率')
    ap.add_argument('--encoder-epochs', type=int, default=100); ap.add_argument('--encoder-steps', type=int, default=15)
    ap.add_argument('--head-epochs', type=int, default=20); ap.add_argument('--head-steps', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--mc-train', type=int, default=2); ap.add_argument('--mc-val', type=int, default=5); ap.add_argument('--mc-eval', type=int, default=20)
    ap.add_argument('--label-smoothing', type=float, default=0.05); ap.add_argument('--kl-beta', type=float, default=0.01)
    ap.add_argument('--prior-std', type=float, default=1.); ap.add_argument('--target-error', type=float, default=0.05)
    args = ap.parse_args(argv)
    if not 0 < args.shrinkage <= 1: ap.error('shrinkage须在(0,1]')
    if not 0 <= args.label_smoothing < 1: ap.error('label-smoothing须在[0,1)')
    if min(args.shots+args.train_shots) < 1 or max(args.train_shots) > 24: ap.error('shot须为正，训练shot最大24')
    if min(args.encoder_lr,args.head_lr,args.finetune_lr,args.prior_std,args.order_max) <= 0: ap.error('学习率/先验标准差/阶次上限须为正')
    if min(args.encoder_epochs,args.head_epochs,args.encoder_steps,args.head_steps,args.episodes,args.mc_train,args.mc_val,args.mc_eval) < 1: ap.error('训练和采样预算须为正')
    if args.kl_beta < 0 or not 0 < args.target_error < 1: ap.error('KL权重非负，目标错误率须在(0,1)')
    return args


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    run(arguments())
