"""GPU gradient, frozen-reference, LOO and source pairing regression tests."""
import copy,json,unittest
from types import SimpleNamespace
import numpy as np
import torch
from prepare_laspi import ROOT
from run_laspi_gfsl import load_samples,seed_all
from run_laspi_cross_702010 import split_cross
from temporal_information import reserve_calibration
from lwc_stability_model import DualLWC,MODES,configure_training,source_objective,support_objective,support_distance_logits
from run_lwc_stability import sample_cross_speed,fixed_memory,state_change

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.cfg=SimpleNamespace(**json.loads((ROOT/'lwc_stability_pilot.json').read_text(encoding='utf-8')));cls.cfg.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def test_sampling(self):
        rows,y,_=load_samples(ROOT/self.cfg.reference_cache,metadata_only=True);role,*_=split_cross(rows,'25hz_0pct',42);role,_=reserve_calibration(rows,y,role,42)
        memory=fixed_memory(rows,y,role,42,20)
        ids,s,q,pairs=sample_cross_speed(memory,rows,y,np.random.default_rng(42),4)
        self.assertEqual(len(ids),40);self.assertTrue(all(role[i]=='base_train' for i in ids));self.assertTrue(all(p['speed_groups'][0]!=p['speed_groups'][1] for p in pairs));self.assertFalse(set(s)&set(q))
    def test_source_and_support_gradients(self):
        cfg=self.cfg;seed_all(42);initial=DualLWC(cfg,'parallel').to(cfg.device)
        x=torch.randn(20,8192,device=cfg.device);labels=torch.arange(5,device=cfg.device).repeat_interleave(4);speeds=torch.tensor([25,25,35,35]*5,device=cfg.device)
        from train_laspi_components import SIGNATURES
        signatures=torch.tensor(SIGNATURES[labels.cpu().numpy()],device=cfg.device);s=np.array([4*c+j for c in range(5) for j in (0,1)]);q=np.array([4*c+j for c in range(5) for j in (2,3)])
        optimizer=configure_training(initial,cfg,'pretrain');loss,info=source_objective(initial,x,labels,signatures,speeds,s,q,cfg,True);loss.backward()
        self.assertTrue(torch.isfinite(initial.discriminative.wavelet.log_scale.grad).all());self.assertGreater(float(initial.stable.wavelet.log_scale.grad.abs().sum()),0);optimizer.step()
        initial.eval();before={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}
        train_labels=torch.tensor(list(range(5))*2+[5,6],device=cfg.device);train_x=torch.randn(12,8192,device=cfg.device);train_sig=torch.tensor(SIGNATURES[train_labels.cpu().numpy()],device=cfg.device);train_speed=torch.tensor([25]*5+[35]*5+[25,35],device=cfg.device)
        for mode in MODES:
            student=copy.deepcopy(initial);student.mode=mode;teacher=copy.deepcopy(student).eval()
            for p in teacher.parameters():p.requires_grad_(False)
            old={k:v.detach().cpu().clone() for k,v in student.stable.state_dict().items()};optimizer=configure_training(student,cfg,'support');student.train();optimizer.zero_grad(set_to_none=True)
            loss,info=support_objective(student,teacher,train_x,train_labels,train_sig,train_speed,cfg,.01*torch.randn_like(train_x));loss.backward();optimizer.step()
            delta=state_change(old,student.stable)
            self.assertEqual(delta==0,mode!='trainable_exchange')
            self.assertEqual(student.discriminative.wavelet.log_scale.grad,None)
            self.assertGreater(float(student.discriminative.components[0].weight.grad.abs().sum()),0)
            if mode in ('frozen_exchange','trainable_exchange'):self.assertGreater(float(student.attention.in_proj_weight.grad.abs().sum()),0)
        self.assertEqual(state_change(before,initial),0)
    def test_loo_vectorization(self):
        torch.manual_seed(3);cfg=self.cfg;labels=torch.tensor(list(range(5))*2+[5,6]);z=torch.nn.functional.normalize(torch.randn(12,64),dim=-1).requires_grad_();reference=torch.randn_like(z)
        score=support_distance_logits(z,reference,labels,cfg);self.assertEqual(tuple(score.shape),(12,7));self.assertTrue(torch.isfinite(score).all());score.sum().backward();self.assertTrue(torch.isfinite(z.grad).all())
        self.assertFalse(reference.requires_grad)

if __name__=='__main__':unittest.main(verbosity=2)
