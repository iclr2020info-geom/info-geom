import torch
import torch.nn as nn
import torch.nn.functional as F
from ops import stiefel, LinearStiefel, LinearOblique
import numpy as np
from tqdm import tqdm, trange
from ops.gradOps import *

tanh = F.tanh
relu = F.relu

"""
Erst tanzte ich mit Mnemosyne, die, glaub' ich,
eine echte Muse war
Dazwischen trank ich Mischgetränke,die hiessen, glaub' ich, Amnesia
Irgendwann fing ich an zu singen,
und ich sang bis die Putzkolonne kam
"""
class BasicBlock(nn.Module):

    def __init__(self, nin, nout, act=tanh, scaling_type=None):
        super(BasicBlock, self).__init__()
        self.fc = nn.Linear(nin, nout)
        self.act = act

    def forward(self, x):
        # TODO: fix this asap
        return self.act(self.fc(x))
        # return self.fc(x)


class StiefelGeneralizedBlock(BasicBlock):

    def __init__(self, nin, nout, act=tanh, scaling_type='isotropic'):
        super(BasicBlock, self).__init__()
        self.fc = LinearStiefel.LinearGeneralizedStiefel(nin, nout, scaling_type='isotropic', bias=True)
        self.act = act


class ObliqueGeneralizedBlock(BasicBlock):

    def __init__(self, nin, nout, act=tanh, scaling_type='isotropic'):
        super(BasicBlock, self).__init__()
        self.fc = LinearOblique.LinearGeneralizedOblique(nin, nout, scaling_type=scaling_type, bias=True)
        self.act = act


class LinearBlock(BasicBlock):

    def __init__(self, nin, nout, scaling_type='isotropic'):
        super(BasicBlock, self).__init__()
        self.fc = nn.Linear(nin, nout)
        act = lambda x : x
        self.act = act


class Net(nn.Module):

    def __init__(self, block, in_width=32**2*3, width=400, layers=200, num_classes=10, sigma_w=1.0151637852088928,
                 sigma_b=0.0021566949867938977, scaling_type='isotropic'):
        super(Net, self).__init__()
        self.p_over_n = np.sqrt(width/in_width)
        self.sigma = sigma_w
        self.step_counter = 0
        self.fc1 = block(in_width, width, scaling_type=scaling_type)
        self.layers = self._make_layer(block, width, layers)
        self.classifier = nn.Linear(in_features=width, out_features=num_classes)
        for m in self.modules():
            if isinstance(m, LinearStiefel.LinearGeneralizedStiefel):
                m.scaling.data.fill_(sigma_w)
            elif isinstance(m,LinearOblique.LinearOblique):
                m.weight.data = stiefel.random(m.weight.size())
                m.scaling.data.fill_(sigma_w)
            elif isinstance(m, nn.Linear):
                m.weight.data = sigma_w * stiefel.random(m.weight.size()[::-1]).t()
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    if sigma_b != 0:
                        m.bias.data.normal_(0, sigma_b)
                    else:
                        m.bias.data.zero_()

    def _make_layer(self, block, width, blocks):
        layers = []
        for i in range(1, blocks):
            layers.append(block(width, width))

        return nn.Sequential(*layers)

    def _covariance(self, x):
        M = x.clone().detach()

        M = M - torch.mean(M, 1, keepdim=True)
        return torch.mm(M, M.t())

    def _jacobian(self, inputs, outputs):
        from torch.autograd import grad
        return torch.stack([grad([outputs[:, i].sum()], [inputs], retain_graph=True, create_graph=True)[0] for i in
                            range(outputs.size(1))], dim=-1)

    def _train(self, optimizer, criterion, train_loader, epoch, cuda, writer=None, run_logger=None):
        self.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Train Batches', ascii=True)):
            if cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target.data).float().mean()

            optimizer.step()
            self.step_counter+=1
            if writer is not None:
                writer.add_scalar('loss_train', loss.data.item()/len(data), self.step_counter)
                writer.add_scalar('accuracy_train', correct.data.item(), self.step_counter)
            if run_logger is not None:
                run_logger.log_scalar("train.loss", loss.data.item()/len(data) )
                run_logger.log_scalar("train.accuracy", correct.data.item())

    def _evaluate(self, criterion, test_loader, epoch, cuda, writer=None, set='Valid'):
        self.eval()
        test_loss = 0
        correct = 0
        samples = 0
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc=set + ' batches', ascii=True)):
            if cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self(data)
                test_loss += criterion(output, target).data  # sum up batch loss
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target.data).sum()
                samples += len(data)
        test_loss /= samples
        accuracy = correct.double() / samples
        if writer is not None:
            writer.add_scalar('accuracy_' + set.lower(), accuracy, epoch * len(test_loader))
            writer.add_scalar('loss_' + set.lower(), test_loss, epoch * len(test_loader))


        return accuracy.item(), test_loss.item()

    def _collect_stats(self, stats_loader, cuda, criterion=None, stats_type='marginal',run_logger=None):
        self.eval()

        if stats_type == 'jacobian':
            data, target = next(iter(stats_loader))
            if cuda:
                data, target = data.cuda(), target.cuda()
            data.requires_grad_()
            stats, _ = self(data, stats_type=stats_type)
            stats = tuple(map(lambda el: el.cpu().detach().numpy(), stats))
            stats = np.dstack(stats)
            stats = np.swapaxes(stats, 0, -1)
            stats = np.linalg.svd(stats, full_matrices=False, compute_uv=False)
        elif stats_type == 'smoothness':
            data, target = next(iter(stats_loader))
            if cuda:
                data, target = data.cuda(), target.cuda()
            x, output = self(data, stats_type=stats_type)
            loss = criterion(output, target)
            # Collect statistics
            lambda_max_Fisher = SmoothnessFisher(loss, output, self)
            lambda_max_AbsHessian = SmoothnessAbsHessian(loss, output, self)
            if run_logger is not None:
                run_logger.log_scalar("curvature.SmoothnessFisher", float(lambda_max_Fisher))
                run_logger.log_scalar("curvature.SmoothnessAbsHessian", float(lambda_max_AbsHessian))
            stats = [lambda_max_Fisher, lambda_max_AbsHessian]
        elif stats_type == 'condition_number':
            data, target = next(iter(stats_loader))
            if cuda:
                data, target = data.cuda(), target.cuda()
            x, output = self(data, stats_type=stats_type)
            loss = criterion(output, target)
            # Collect statistics
            lambda_max_AbsHessian = SmoothnessAbsHessian(loss, output, self)
            lambda_min_AbsHessian = StrongConvexity(loss, output, self, lambda_max=lambda_max_AbsHessian)
            if run_logger is not None:
                run_logger.log_scalar("curvature.SmoothnessAbsHessian", float(lambda_max_AbsHessian))
                run_logger.log_scalar("curvature.SCAbsHessian", float(lambda_min_AbsHessian))
                run_logger.log_scalar("curvature.ConditionNumber", float(lambda_max_AbsHessian/ lambda_min_AbsHessian))
            stats = [lambda_max_AbsHessian, lambda_min_AbsHessian]
        else:
            stats_temp = []
            stats = []
            for batch_idx, (data, target) in enumerate(tqdm(stats_loader, desc='Stat batches', ascii=True)):
                data, target = data.cuda(), target.cuda()
                stat, _ = self(data, stats_type=stats_type)
                stats_temp += stat
            stats_temp = np.hstack(stats_temp)
            stats_temp = np.swapaxes(stats_temp, 1, 2)
            import transplant
            matlab = transplant.Matlab()
            matlab.addpath("/home/***/projects/isonetry/projectionpursuit")
            for i in range(stats_temp.shape[0]):
                _, projdata, _ = matlab.NGCA(stats_temp[0].astype(np.double), transplant.MatlabStruct({"nbng": 1, "sphered_proj": 0}))
                stats += [projdata]
            matlab.exit()
            stats = np.vstack(stats)
        return stats

    def forward(self, x, stats_type=None):
        x = x.view(x.size(0), -1)

        if stats_type is None:

            x = self.fc1(x)

            x = self.layers(x)

            x = self.classifier(x)

            return x

        elif stats_type == 'cov':
            covmats = []
            covmats += [self._covariance(x)]

            x = self.fc1(x)
            covmats += [self._covariance(x)]

            for name, m in self.layers._modules.items():
                x = m(x)
                covmats += [self._covariance(x)]

            x = self.classifier(x)
            return covmats, x
        elif stats_type == 'jacobian':
            x1 = self.fc1(x)
            x1 = self.layers(x1)

            J = self._jacobian(x, x1)
            return J, x1
        elif stats_type == 'marginal':
            marginal = []
            marginal += [self.fc1.fc(x).cpu().detach().numpy()]
            x = self.fc1(x)
            for layer, (name, m) in enumerate(self.layers._modules.items()):
                if layer in {99, 198}:
                    marginal += [m.fc(x).cpu().detach().numpy()]
                x = m(x)
            x = self.classifier(x)
            marginal = [np.array(marginal)]
            return marginal, x
        elif stats_type == 'condition_number' or stats_type == 'smoothness':
            x = self.fc1(x)

            x = self.layers(x)

            z = self.classifier(x)

            return x, z


    def register_optimizer(self):
        # list of dicts containing the params of interest grouped into manifolds
        stiefel_params = []
        oblique_params = []
        scaling_params = []

        for m in self.modules():
            if isinstance(m, LinearStiefel.LinearStiefel):
                stiefel_params.append(m.weight)
            elif isinstance(m, LinearOblique.LinearOblique):
                oblique_params.append(m.weight)
            if isinstance(m,(LinearOblique.LinearOblique, LinearStiefel.LinearStiefel)):
                scaling_params.append(m.scaling)

        stiefel_params = list(map(id, stiefel_params))
        oblique_params = list(map(id, oblique_params))
        scaling_params = list(map(id, scaling_params))
        ep = filter(lambda p: id(p) not in stiefel_params + oblique_params + scaling_params,
                             self.parameters())
        sp = filter(lambda p: id(p) in stiefel_params, self.parameters())
        scp = filter(lambda p: id(p) in scaling_params, self.parameters())
        op = filter(lambda p: id(p) in oblique_params, self.parameters())

        return [{'params': ep, 'manifold': 'euclidean'},
                {'params': sp, 'manifold': 'stiefel'},
                {'params': op, 'manifold': 'oblique'},
                {'params': scp, 'manifold': 'scaling'}]