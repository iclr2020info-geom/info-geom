import torch
import numpy as np
from torch.nn.functional import normalize as normalize
r"""
Inspired by Riemannian approach to batch normalization
https://github.com/MinhyungCho/riemannian-batch-normalization-pytorch
"""
def innerproduct(X, Z):
    pass


def project(X, Z):
    r"""

    :param X: point in the manifold
    :param Z: point in the ambient space
    :return:  point on the tangent at X
    """
    return Z - X * torch.sum(X * Z, dim=1, keepdim=True)


def normal(X,Z):
    pass


def exp(X,Z):
    r"""
    Uses exponential map rather than retraction.
    :param X: parameter, point in the manifold
    :param Z: eta* \pi(\nabla f(X))
    :return Y:
    """
    norm = Z.norm(p=2, dim=1, keepdim=True)
    Z = normalize(Z, dim=1, p=2, eps=1E-8)
    # mask = norm < 0.5
    # mask = torch.squeeze(torch.nonzero(torch.squeeze(mask)))
    Y = X*norm.cos() + Z*norm.sin()
    # Y[:, mask] = _normalize_columns(X[:,mask] + Z[:,mask])
    # Y[:, mask] = normalize( X[:,mask] + Z[:,mask], dim=1, p=2, eps=1E-8)
    return Y


def transport(X,Z, Delta=None):
    if Delta is None:
        norm = Z.norm(p=2, dim=1, keepdim=True)
        u = normalize(Z, dim=1, p=2, eps=1E-8)
        return (u*norm.cos() - X*norm.sin())*norm
    else:
        norm = Z.norm(p=2, dim=1, keepdim=True)
        u = normalize(Z, dim=1, p=2, eps=1E-8)

        return Delta - (u*(1-norm.cos()) + X*norm.sin()) * torch.sum(u * Delta, dim=0, keepdim=True)


def typical_dist(X):
    return torch.from_numpy(np.pi)*torch.sqrt(X.size()[1])


def randomvec(size,X):
    r"""

    :param size:
    :param X: point in the manifold
    :return: unit 2-norm point on the tangent at X
    """
    Z = torch.randn(size)
    Z = project(X,Z)
    return Z/Z.norm(p=2)


def random(size):
    X = torch.randn(size)
    return normalize(X, dim=1 ,p=2, eps=1E-8)


def _normalize_columns(X):
    return X.div_(X.norm(dim=0, p=2, keepdim=True))