import sys, os
from models.models import Net, BasicBlock, ObliqueGeneralizedBlock, StiefelGeneralizedBlock, LinearBlock
import torch
import torch.nn as nn
import ops.optim as optim
from tqdm import trange
import os
import random
from random_words import RandomWords
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from ops.computeParams import optimize_sigmaw_sigmab
from sacred.settings import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'

sys.path.insert(0, os.path.abspath('.'))
rw = RandomWords()
ex = Experiment(name=rw.random_word() + str(random.randint(0, 100)))
# ex.observers.append(MongoObserver.create(db_name='MOREexperiments'))

ex.observers.append(MongoObserver(
    url='mongodb://***:***@***.***.com/the_database?authMechanism=SCRAM-SHA-1',
    db_name='MOREexperiments'))

@ex.pre_run_hook
def set_logger_stream(_run):
    _run.root_logger.handlers[0].stream = sys.stderr

@ex.named_config
def oblique_penalized():
    manifold = 'oblique_penalized'
    learning_rate_euclidean = 0.000027057413841861918
    learning_rate_manifold = 0.000016212951529473764
    learning_rate_scale = 0.00006090393338671865
    omega = 0.0007142493569943361
    weight_decay = 0.01

@ex.named_config
def stiefel_penalized():
    manifold = 'stiefel_penalized'
    learning_rate_euclidean = 0.00010000000000000009
    learning_rate_manifold  = 0.000012978120652662701
    learning_rate_scale     = 0.00009847921206987803
    weight_decay = 0.01

@ex.named_config
def stiefel2():
    manifold = 'stiefel'
    learning_rate_euclidean = 0.00010000000000000009
    learning_rate_manifold  = 0.000012978120652662701
    learning_rate_scale     = 0.00009847921206987803


@ex.named_config
def oblique():
    manifold = 'oblique'
    learning_rate_euclidean = 0.000027057413841861918
    learning_rate_manifold = 0.000016212951529473764
    learning_rate_scale = 0.00006090393338671865
    omega = 0.0007142493569943361


@ex.config
def cfg():
    h0 = 1/64
    sigma_w, sigma_b = optimize_sigmaw_sigmab(h0)
    batch_size = 1000
    learning_rate_euclidean = 0.000009729809981695022
    learning_rate_manifold = 0.
    learning_rate_scale = 0.
    manifold = 'euclidean'
    cuda = True
    dataset = 'CIFAR10'
    data_dir = '/home/***/projects/isonetry/data'
    epochs = 200
    omega = 0
    gpu = 0
    weight_decay = 0

def get_data(dataset, cuda, batch_size,_seed, h0, data_dir):
    from ops.transformsParams import CIFAR10, SVHN
    from torchvision import datasets, transforms
    from ops.utils import SubsetSequentialSampler

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR10':
        transform_train, transform_eval = CIFAR10(h0)
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        stats_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_eval)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_eval)
    elif dataset == 'SVHN':
        transform_train, transform_eval = SVHN(h0)
        train_set = datasets.SVHN(root=data_dir, split='train', download=True,
                                  transform=transform_train)
        test_set = datasets.SVHN(root=data_dir, split='test', download=True,
                                 transform=transform_eval)
        stats_set = datasets.SVHN(root=data_dir, split='train', download=True,
                                 transform=transform_eval)

    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.seed(_seed)
    np.random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    stats_idx = np.array(indices)[np.random.choice(len(indices), 30)]
    stats_idx = stats_idx.tolist()

    stats_sampler = SubsetSequentialSampler(stats_idx)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    stats_loader = torch.utils.data.DataLoader(stats_set, batch_size=30, sampler=stats_sampler)

    return train_loader, test_loader, stats_loader

@ex.automain
def run(batch_size,
        learning_rate_euclidean,
        learning_rate_manifold,
        learning_rate_scale,
        omega,
        epochs,
        manifold,
        dataset,
        cuda,
        gpu,
        data_dir,
        sigma_w,
        sigma_b,
        h0,
        weight_decay,
        _run,
        _seed):


    writer = None
    with torch.cuda.device(gpu):
        torch.cuda.manual_seed(_seed) if cuda is True else torch.manual_seed(_seed)

    if 'euclidean' in manifold:
        model = Net(BasicBlock, sigma_w=sigma_w, sigma_b=sigma_b)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_euclidean)
    elif 'linear' in manifold:
        model = Net(LinearBlock, sigma_w=1, sigma_b=0)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_euclidean)
    elif 'stiefel' in manifold:
        model = Net(StiefelGeneralizedBlock, sigma_w=sigma_w, sigma_b=sigma_b)
        params = model.register_optimizer()
        for d in params:
            if d['manifold'] is 'stiefel':
                d.update({'lr': learning_rate_manifold})
            elif d['manifold'] is 'scaling':
                d.update({'manifold': 'euclidean'})
                d.update({'lr': learning_rate_scale})
                d.update({'weight_decay': weight_decay})
        optimizer = optim.Adam(params, lr=learning_rate_euclidean)
    elif 'oblique' in manifold:
        model = Net(ObliqueGeneralizedBlock, sigma_w=sigma_w, sigma_b=sigma_b)
        params = model.register_optimizer()
        for d in params:
            if d['manifold'] is 'oblique':
                d.update({'omega': omega})
                d.update({'lr': learning_rate_manifold})
            elif d['manifold'] is 'scaling':
                d.update({'manifold': 'euclidean'})
                d.update({'lr': learning_rate_scale})
                d.update({'weight_decay': weight_decay})
        optimizer = optim.Adam(params, lr=learning_rate_euclidean)

    criterion = nn.CrossEntropyLoss()
    with torch.cuda.device(gpu):
        if cuda is True:
            model.cuda(gpu)

        train_loader, test_loader, stats_loader = get_data(dataset, cuda, batch_size, _seed, h0=h0, data_dir=data_dir)
        for epoch in trange(1, epochs+1, desc=' Epochs', ascii=True):
            if epoch in set(range(5, 50, 5)).union({1}).union(set(range(100, 200, 25))):
                J = model._collect_stats(stats_loader, cuda, stats_type='jacobian')
                _run.log_scalar("sigma.max.jac", float(np.max(J)))

            model._collect_stats(test_loader, cuda, criterion=criterion, stats_type='smoothness', run_logger=_run)
            model._train(optimizer, criterion, train_loader, epoch, cuda, writer, run_logger=_run)
            test_accuracy, test_loss = model._evaluate(criterion, test_loader, epoch, cuda, writer, set='Test')
            #Sacred stat logger
            _run.log_scalar("test.accuracy", test_accuracy)
            _run.log_scalar("test.loss", test_loss)

        print('Test loss: {:0.4f} \n'.format(test_loss,))
        print('Test accuracy: {0:.1f}% \n'.format(test_accuracy*100))
