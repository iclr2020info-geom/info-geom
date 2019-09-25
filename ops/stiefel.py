import torch

mml = torch.matmul
tr = torch.trace
diag = torch.diag
sign = torch.sign


def innerproduct(X1, X2):
    """Returns the Euclidean inner product between matrices X1 and X2 belonging to a manifold embedded in the Euclidean manifol

    .. math ::
        \langle X_1 , X_2 \rangle = \textrm{tr}[X_1^T X_2]

    Args:
        X1 (Variable): First input.
        X2 (Variable): Second input (of size matching x1).

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.innerproduct(input1, input2)
        >>> print(output)
    """
    return tr(mml(torch.t(X1), X2))


def sym(X):
    r""" Creates a symmetric matrix from the input

    .. math::
    \textrm{sym}(X) = \frac{1}{2}\left(X^T + X \right)

    Args:
        X (Variable): Square matrix
    """

    return 0.5 * (X + X.t())


def skew(X):
    r""" Creates a skew-symmetric matrix from the input

    .. math::
    \textrm{skew}(X) = \frac{1}{2}\left(X - X^T \right)

    Args:
        X (Variable): Square matrix
    """
    return 0.5 * (X - X.t())


def project(X,Z):
    XtZ = mml(X.t(),Z)
    symXtZ = sym(XtZ)
    return Z.sub_(mml(X,symXtZ))

def normal(X, Z):
    r""" Projects Z onto normal at X
    .. math::
    \pi(Z) = X \textrm{skew}\left(X^TZ\right)+\left( I - XX^T \right)Z
    TODO: fix math description
    """
    return mml(X, sym( mml(X.t(), Z)) )

def retraction(Z):
    if False: # TODO: remove this - old code was sped up by running the matrix factorization on the CPU but the new
        # pytorch code is much faster doing it on the CPU
        Z = Z.cpu()
        Q, R = torch.qr(Z)
        R = diag(sign(sign(diag(R)) + .5))
        return mml(Q, R).cuda(non_blocking=True)
    else:
        Z = Z
        Q, R = torch.qr(Z)
        R = diag(sign(sign(diag(R)) + .5))
        return mml(Q, R)


def transport(X, Z):
    r""" Translates vector Z parallel to the geodesic spanned by X_{t-1} and X_t where both X_{t-1} and X_t belong to
    the manifold.

    Warning: Z is assumed to be the tangent at X_{t-1}, and X_t is assumed to be the value of X_{t-1} after gradient
    update using the retraction.

    Args:
    X (Variable): Matrix belonging to manifold.
    Z (Variable): Tangent matrix.
    ptZ (Variable): Tangent matrix at X.
    """
    return project(X,Z)


def typical_dist(X):
    r"""

    :param X:
    :return: sqrt( p * k )
    """

    return torch.sqrt(X.size()[0]*X.size()[1])


def random(size):
    q, _ = torch.qr(torch.randn(size[0], size[1]))
    return q
