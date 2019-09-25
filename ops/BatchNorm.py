import torch
import torch.nn as nn
from torch.autograd import Variable

class _MeanOnlyBatchNorm(nn.Module):

    def __init__(self, num_features,momentum=0.1):
        super(_MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        # self.running_mean = Parameter(torch.Tensor(num_features, 1))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()

    def forward(self, input):

        # mu = Variable(torch.mean(input,dim=0, keepdim=True).data, requires_grad=False)
        if self.training is True:
            mu = input.mean(dim=0, keepdim=True)
            mu = self.momentum * mu + (1 - self.momentum) * Variable(self.running_mean)

            self.running_mean = mu.data
            return input.sub_(mu)
        else:
            return  input.sub_(Variable(self.running_mean))

    def __repr__(self):
        return ('{name}({num_features},momentum={momentum})'
                .format(name=self.__class__.__name__, **self.__dict__))
class BatchNorm1d(_MeanOnlyBatchNorm):
    r"""Applies Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::
Ń
        y = x - mean[x]

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable
            affine parameters. Default: True

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm1d, self)._check_input_dim(input)