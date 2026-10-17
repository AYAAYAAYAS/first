"""One shared, shrinkage full-covariance Mahalanobis metric for train and test."""
import torch


def fit_reference(support, labels, classes=5, shrinkage=.2, detach_covariance=True):
    """Only reference/support features fit covariance, never query features."""
    means=torch.stack([support[labels==c].mean(0) for c in range(classes)])
    residual=support-means[labels]
    covariance=residual.T@residual/max(len(support)-classes,1)
    if detach_covariance:covariance=covariance.detach()
    dimension=support.shape[-1]
    scale=torch.diagonal(covariance).mean().clamp_min(1e-5)
    covariance=(1-shrinkage)*covariance+shrinkage*scale*torch.eye(dimension,device=support.device,dtype=support.dtype)
    cholesky=torch.linalg.cholesky(covariance)
    return means,cholesky


def whiten(features,cholesky):
    return torch.linalg.solve_triangular(cholesky,features.T,upper=False).T


def mahalanobis_logits(features,means,cholesky):
    z=whiten(features,cholesky);m=whiten(means,cholesky)
    squared=(z.square().sum(-1)[:,None]+m.square().sum(-1)[None]-2*z@m.T).clamp_min(0.)
    return -squared/features.shape[-1]
