"""Mean-field Gaussian weight VI and categorical posterior uncertainty."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    def __init__(self,incoming,outgoing,bayesian=True,prior_std=1.,initial_std=.02):
        super().__init__();self.bayesian=bayesian;self.prior_std=prior_std
        self.weight_mu=nn.Parameter(torch.empty(outgoing,incoming));self.bias_mu=nn.Parameter(torch.empty(outgoing))
        nn.init.kaiming_uniform_(self.weight_mu,a=math.sqrt(5));nn.init.uniform_(self.bias_mu,-1/math.sqrt(incoming),1/math.sqrt(incoming))
        rho=math.log(math.expm1(initial_std))
        self.weight_rho=nn.Parameter(torch.full_like(self.weight_mu,rho),requires_grad=bayesian)
        self.bias_rho=nn.Parameter(torch.full_like(self.bias_mu,rho),requires_grad=bayesian)
        self.weight_sample=None;self.bias_sample=None
    def resample(self,sample=True):
        if self.bayesian and sample:
            self.weight_sample=self.weight_mu+F.softplus(self.weight_rho)*torch.randn_like(self.weight_mu)
            self.bias_sample=self.bias_mu+F.softplus(self.bias_rho)*torch.randn_like(self.bias_mu)
        else:self.weight_sample=self.weight_mu+0.;self.bias_sample=self.bias_mu+0.
    def forward(self,x):
        if self.weight_sample is None:self.resample()
        return F.linear(x,self.weight_sample,self.bias_sample)
    def kl(self):
        if not self.bayesian:return self.weight_mu.new_zeros(())
        total=0.
        for mu,rho in ((self.weight_mu,self.weight_rho),(self.bias_mu,self.bias_rho)):
            variance=F.softplus(rho).square().clamp_min(1e-12)
            total+=.5*(variance/self.prior_std**2+mu.square()/self.prior_std**2-1+
                       math.log(self.prior_std**2)-variance.log()).sum()
        return total


class FeatureHead(nn.Module):
    def __init__(self,bayesian=True,prior_std=1.):
        super().__init__()
        self.layers=nn.ModuleList([BayesianLinear(a,b,bayesian,prior_std) for a,b in ((128,50),(50,50),(50,32))])
    def resample(self,sample=True):
        for layer in self.layers:layer.resample(sample)
    def forward(self,x):
        for layer in self.layers[:-1]:x=torch.tanh(layer(x))
        return F.normalize(self.layers[-1](x),dim=-1)
    def kl(self):return sum(layer.kl() for layer in self.layers)
    def count(self):return sum(layer.weight_mu.numel()+layer.bias_mu.numel() for layer in self.layers)


def uncertainty(probabilities):
    p=np.asarray(probabilities,dtype=np.float64).clip(1e-12);p/=p.sum(-1,keepdims=True)
    mean=p.mean(0);entropy=-(mean*np.log(mean)).sum(-1)
    expected=-(p*np.log(p)).sum(-1).mean(0)
    return mean,entropy,expected,np.maximum(entropy-expected,0.)
