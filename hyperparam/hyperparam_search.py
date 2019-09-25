import sys, os
sys.path.insert(0, os.path.abspath('.'))
from models.models import Net, BasicBlock, ObliqueGeneralizedBlock, StiefelGeneralizedBlock, LinearBlock
import torch
import torch.nn as nn
import ops.optim as optim
from tqdm import trange
import random
from random_words import RandomWords
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from ops.computeParams import optimize_sigmaw_sigmab
from sacred.settings import SETTINGS
# SETTINGS.CAPTURE_MODE = 'sys'

rw = RandomWords()
ex = Experiment(name=rw.random_word() + str(random.randint(0,100)))
# ex.observers.append(MongoObserver.create(db_name='isonetry-hyperparams2'))
ex.observers.append(MongoObserver.create(
    url='mongodb://***:***@***.***.com/admin?authMechanism=SCRAM-SHA-1',
    db_name='isontery-hyperparams2'))

@ex.pre_run_hook
def set_logger_stream(_run):
    _run.root_logger.handlers[0].stream = sys.stderr
@ex.named_config
def stiefel_penalized():
    manifold = 'stiefel_penalized'
    learning_rate_manifold = 0.001
    weight_decay = 0.01

def oblique_penalized():
    manifold = 'oblique_penalized'
    learning_rate_manifold = 0.001
    weight_decay = 0.01

@ex.named_config
def stiefel():
    manifold = 'stiefel'
    learning_rate_manifold = 0.001

@ex.named_config
def oblique():
    manifold = 'oblique'
    learning_rate_euclidean = 0.0009527434244085628
    learning_rate_manifold = 0.02839317856412252
    learning_rate_scale = 0.04967936914564948
    omega = 0.00029289094449421306



@ex.config
def cfg():
    h0 = 1/64
    sigma_w, sigma_b = optimize_sigmaw_sigmab(h0)
    batch_size = 1000
    learning_rate_euclidean = 0.012
    learning_rate_manifold = 0.1
    learning_rate_scale = 0.1
    manifold = 'euclidean'
    cuda = True
    dataset = 'CIFAR10'
    data_dir = '/home/***/logs/'
    epochs = 20
    omega = 0
    gpu = 0
    samples_train = 40_000
    weight_decay = 0


def get_data(dataset, cuda, batch_size,_seed, samples_train, h0, data_dir):
    from ops.transformsParams import CIFAR10, SVHN
    from torchvision import datasets, transforms

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
        test_set = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    elif dataset == 'CIFAR10':
        transform, transform_eval = CIFAR10(h0)
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_eval)
    elif dataset == 'SVHN':
        transform, transform_eval = SVHN(h0)
        train_set = datasets.SVHN(root=data_dir, split='train', download=False,
                                  transform=transform)
        test_set = datasets.SVHN(root=data_dir, split='test', download=False,
                                 transform=transform_eval)
    split = 10_000
    split2 = 1_000 # one minibatch to test progress
    # split2 < split
    assert split2 < split
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.seed(_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx, valid_small_idx = indices[split:split+samples_train], indices[:split], indices[:split2]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
    valid_small_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_small_idx)

    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler,
                                               shuffle=False, **kwargs)
    valid_small_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=valid_small_sampler,
                                               shuffle=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                                **kwargs)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, valid_small_loader, test_loader



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
        samples_train,
        weight_decay,
        _run,
        _seed):

    writer = None
    #seed
    with torch.cuda.device(gpu): torch.cuda.manual_seed(_seed) if cuda is True else torch.manual_seed(_seed)

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

        # the data, shuffled and split between train and test sets
        train_loader, valid_loader, valid_small_loader, test_loader =\
            get_data(dataset, cuda, batch_size, _seed, samples_train=samples_train,  h0=h0, data_dir=data_dir)
        for epoch in trange(1, epochs+1, desc=' Epochs', ascii=True):
            model._train(optimizer, criterion, train_loader, epoch, cuda, writer, run_logger=_run)
            valid_accuracy, valid_loss = model._evaluate(criterion, valid_small_loader, epoch, cuda, writer)
            #Sacred stat logger
            _run.log_scalar("valid.accuracy", valid_accuracy)
            _run.log_scalar("valid.loss", valid_loss)

        valid_accuracy, valid_loss =  model._evaluate(criterion, valid_loader, epoch, cuda, writer)
        _run.log_scalar("valid.accuracy", valid_accuracy)
        _run.log_scalar("valid.loss", valid_loss)
        test_accuracy, test_loss = model._evaluate(criterion, test_loader, epoch, cuda, writer, set='Test')
        _run.log_scalar("test.accuracy", test_accuracy)
        _run.log_scalar("test.loss", test_loss)
        print('Test loss: {:0.4f} \nValidation loss: {:0.4f}'.format(test_loss,valid_loss))
        print('Test accuracy: {0:.1f}% \nValid accuracy {0:.1f}%'.format(test_accuracy*100, valid_accuracy*100))

    results = valid_accuracy
    return results
