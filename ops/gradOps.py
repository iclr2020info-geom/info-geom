import numpy as np
import torch

def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def vector_to_parameter_list(vec, parameters):
    r"""Convert one vector to the parameter list

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None
    params_new = []
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param_new = vec[pointer:pointer + num_param].view_as(param).data
        params_new.append(param_new)
        # Increment the pointer
        pointer += num_param

    return list(params_new)


def Rop(ys, xs, vs):
    if isinstance(ys, tuple):
        ws = [torch.tensor(torch.zeros_like(y), requires_grad=True) for y in ys]
    else:
        ws = torch.tensor(torch.zeros_like(ys), requires_grad=True)

    gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)
    re = torch.autograd.grad(gs, ws, grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=True)
    return tuple([j.detach() for j in re])


def Lop(ys, xs, ws):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)
    return tuple([j.detach() for j in vJ])


def HesssianVectorProduct(f, x, v):
    df_dx = torch.autograd.grad(f, x, create_graph=True, retain_graph=True)
    Hv = Rop(df_dx, x, v)
    return tuple([j.detach() for j in Hv])


def HessianSqrVectorProduct(f, x, v):
    #     vp = torch.nn.utils.vector_to_parameters(vp, model.parameters())
    Hv = HesssianVectorProduct(f, x, v)
    HHv = HesssianVectorProduct(f, x, Hv)
    return tuple([j.detach() for j in HHv])


def FisherVectorProduct(loss, output, model, vp):
    #     vp = torch.nn.utils.vector_to_parameters(vp, model.parameters())

    Jv = Rop(output, list(model.parameters()), vp)
    batch, dims = output.size(0), output.size(1)
    if loss.grad_fn.__class__.__name__ == 'NllLossBackward':
        outputsoftmax = torch.nn.functional.softmax(output, dim=1)
        M = torch.zeros(batch, dims, dims).cuda() if outputsoftmax.is_cuda else torch.zeros(batch, dims, dims)
        M.reshape(batch, -1)[:, ::dims + 1] = outputsoftmax
        H = M - torch.einsum('bi,bj->bij', (outputsoftmax, outputsoftmax))
        HJv = [torch.squeeze(H @ torch.unsqueeze(Jv[0],
                                                 -1)) / batch]  # can't figure out why but the n-mode product scales everything up by # batch
    else:
        HJv = HesssianVectorProduct(loss, output, Jv)
    JHJv = Lop(output, list(model.parameters()), HJv)

    # return torch.cat([torch.flatten(v) for v in JHJv]) / batch
    return torch.cat([torch.flatten(v) for v in JHJv])


def SmoothnessFisher(loss, output, model):
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.linalg import LinearOperator

    ndim = len(torch.nn.utils.parameters_to_vector(model.parameters()))

    def NumpyWrapper(vp):
        vp = vector_to_parameter_list(torch.tensor(np.squeeze(vp), device='cuda').float(), model.parameters())
        JHJv = FisherVectorProduct(loss, output, model, vp)
        return JHJv.cpu().numpy()

    A = LinearOperator((ndim, ndim), matvec=NumpyWrapper)
    lambda_max = eigsh(A, 1, which='LM', return_eigenvectors=0, tol=1e-2)
    return lambda_max


def SmoothnessAbsHessian(loss, output, model):
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.linalg import LinearOperator

    ndim = len(torch.nn.utils.parameters_to_vector(model.parameters()))
    batch = output.size(0)

    def NumpyWrapper(vp):
        vp = vector_to_parameter_list(torch.tensor(np.squeeze(vp), device='cuda').float(), model.parameters())
        Hv = HessianSqrVectorProduct(loss, list(model.parameters()), vp)
        Hv = torch.cat([torch.flatten(v) for v in Hv])
        return Hv.cpu().numpy()

    A = LinearOperator((ndim, ndim), matvec=NumpyWrapper)
    lambda_max = eigsh(A, 1, which='LM', return_eigenvectors=0, tol=1e-3)
    return np.sqrt(lambda_max)


def StrongConvexity(loss, output, model, lambda_max=None):
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.linalg import LinearOperator, lobpcg

    # SmoothnessAbsHessian returns the maximum eigenvalue of the absolute Hessian (i.e. the square root of the maximum of the squared Hessian)
    if lambda_max is None:
        lambda_max = SmoothnessAbsHessian(loss, output, model)

    ndim = len(torch.nn.utils.parameters_to_vector(model.parameters()))
    batch = output.size(0)

    def NumpyWrapper(vp):
        vp = vector_to_parameter_list(torch.tensor(np.squeeze(vp), device='cuda').float(), model.parameters())
        Hv = HessianSqrVectorProduct(loss, list(model.parameters()), vp)
        Hv = torch.cat([torch.flatten(v) for v in Hv]) / batch
        return Hv.cpu().numpy()

    A = LinearOperator((ndim, ndim), matvec=NumpyWrapper)
    # use the Lanczos method to estimate the largest Largest (in magnitude) eigenvalue ('LM')
    lambda_min, vs = eigsh(A, 1, which='LM', return_eigenvectors=1, tol=1e-2, sigma=lambda_max**2)
    l,v = lobpcg(A, vs, B=None, M=None, Y=None, tol=None, maxiter=50, largest=False, verbosityLevel=0, retLambdaHistory=False, retResidualNormsHistory=False)

    return np.sqrt(l)