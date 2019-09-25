import math
from torch.optim import Optimizer
from torch.optim.optimizer import required
from ops import stiefel
from ops import oblique
import torch
from torch import mm as mm
r"""
Assortment of stochastic gradient manifold optimizers. Currently only adapted to matricial arguments. To adapt to, say, 
tensor valued weights, explicit matricization has to be performed at the beginning of the inner loop, 
for both weights and the gradients.
"""

class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0,  manifold='euclidean', nu=0.1, omega=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, manifold=manifold, nu=nu, omega=omega)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group['manifold'] is 'euclidean':
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    if group['weight_decay'] != 0:
                        grad = grad.add(group['weight_decay'], p.data)

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)
            elif group['manifold'] is 'stiefel':
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    grad = stiefel.project(p.data,grad)
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    if group['weight_decay'] != 0:
                        grad = grad.add(group['weight_decay'], p.data)

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data = stiefel.retraction(p.data)
                    exp_avg = stiefel.transport(p.data, exp_avg)
            elif group['manifold'] is 'oblique':
                omega = group['omega']
                nu = group['nu']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data.copy_( oblique.normalize(p.data) ) #TODO: look up why Cho & Lee do this
                    grad = p.grad.data
                    grad = oblique.project(p.data,grad)
                    if omega != 0:
                        grad.add_(2*omega, _ortho_penalty(p.data))
                    if nu is not None:
                        grad = _norm_clip(grad, nu)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # if group['weight_decay'] != 0:
                    #     grad = grad.add(group['weight_decay'], p.data)

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    d = -step_size*exp_avg/denom
                    exp_avg = oblique.transport(p.data, d, exp_avg)
                    p.data = oblique.exp(p.data, d)


        return loss


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, manifold='euclidean',nu=0.1, omega=0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, manifold=manifold, nu=nu, omega=omega)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            if group['manifold'] is 'euclidean':
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)
            elif group['manifold'] is 'stiefel':
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    d_p = stiefel.project(p.data,d_p)

                    # if weight_decay != 0:
                    #     d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)
                    p.data = stiefel.retraction(p.data)
                    if momentum!=0:
                        buf = stiefel.transport(p.data, buf)
            elif group['manifold'] is 'oblique':
                omega = group['omega']
                nu = group['nu']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data.copy_(oblique.normalize(p.data)) #TODO: look up why Cho & Lee do
                    d_p = p.grad.data
                    d_p = oblique.project(p.data, d_p)

                    # if weight_decay != 0:
                    #     d_p.add_(weight_decay, p.data)
                    if omega != 0:
                        d_p.add_(2*omega, _ortho_penalty(p.data) )
                    if nu is not None:
                        d_p = _norm_clip(d_p, nu)

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    # p.data.add_(-group['lr'], d_p)
                    if momentum!=0:
                        buf = oblique.transport(p.data, -group['lr'] * d_p)
                    p.data = oblique.exp(p.data, -group['lr']*d_p)

        return loss


def _norm_clip(Z,nu):
    p = 2
    dim = 0
    return torch.renorm(Z, p, dim, nu)


def _ortho_penalty(X):
    r"""
    Computes the gradient of
    .. math::
    ||X'TX - I||_{Fro}
    :param X: weight matrix
    :return:
    """
    #TODO: fix math note about ortho penalty
    return mm( mm(X, X.t()), X ) - X
def _smooth_abs(X, k=5):
    return 2/k * torch.log(1 + torch.exp(X*k)) - X - 2/k*torch.log(2)
def _inv_smooth_abs(X, k=5):
    #TODO: implement inverse function
    pass
