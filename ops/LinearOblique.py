import torch
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from ops import oblique


class LinearOblique(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = A'x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: False

    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`

    Attributes:
        weight: the learnable weights of the module of shape
            (in_features x out_features ).

        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = LinearStiefel(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """
    def __init__(self, in_features, out_features, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return F.linear(input, torch.t(self.weight), self.bias)

    def reset_parameters(self):
        self.weight.data = oblique.random(self.weight.size())
        if self.bias is not None:
            self.bias.data.zero_()


class LinearGeneralizedOblique(LinearOblique):
    r"""Applies a linear transformation to the incoming data: :math:`y = A'x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: False

    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`

    Attributes:
        weight: the learnable weights of the module of shape
            (in_features x out_features ).
        scaling: a learnable scaling parameter, that scales the weight matrix

        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = LinearStiefel(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, scaling_type='isotropic', bias=False):
        r"""

        :param in_features:
        :param out_features:
        :param scaling_type: 'isotropic' or 'anisotropic'.  In the latter case it multiplies the  Stiefel weights with
                             a PSD diagonal matrix, while the former uses a non-negative real value.
        :param bias:
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if scaling_type is 'isotropic':
            self.scaling = Parameter(torch.Tensor(1))
        elif scaling_type is 'anisotropic':
            self.scaling = Parameter(torch.Tensor(min(in_features, out_features)))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return F.linear(input, torch.t(torch.abs(self.scaling) * self.weight), self.bias)

    def reset_parameters(self):
        self.weight.data = oblique.random(self.weight.size())
        self.scaling.data.fill_(1)
        if self.bias is not None:
            self.bias.data.zero_()