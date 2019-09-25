import sys, os
from socket import gethostname
from hyperparam_search import ex
import random
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import Optimizer
from sklearn.externals.joblib import Parallel, delayed
from random_words import RandomWords
import tweepy

dataset = sys.argv[1]
manifold = sys.argv[2]
h0 = float(sys.argv[3])

if gethostname!='***':
    from sshtunnel import SSHTunnelForwarder
    MONGO_HOST = "***.***.com"
    MONGO_USER = "***"
    MONGO_PASS = "***"
    server = SSHTunnelForwarder(
        MONGO_HOST,
        ssh_username=MONGO_USER,
        ssh_password=MONGO_PASS,
        remote_bind_address=('127.0.0.1', 27017),
        local_bind_address=('127.0.0.1', 27017)
    )
    server.start()
    # client = MongoClient('127.0.0.1', server.local_bind_port)
    GPU = 0
    if gethostname() == '***':
        project_dir = '/***/home/***/projects'
    else:
        project_dir = 'home/***/isonetry'
    data_dir = os.path.join(project_dir, 'data')
else:
    from ops.utils import bestGPU
    GPU = bestGPU()
    project_dir = '/home/***/projects/isonetry/'
    data_dir = os.path.join(project_dir,'data')

consumer_key = "***"
consumer_secret = "***"
access_token = "***"
access_token_secret = "***"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

rw = RandomWords()
randname = rw.random_word() + str(random.randint(0, 100))

if manifold == 'stiefel':
    space = [Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_euclidean'),
             Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_manifold'),
             Real(10 ** -8, 10 ** -3, "log-uniform", name='learning_rate_scale')]
             # Real(1, 100, "log-uniform", name='omega')
    @use_named_args(space)
    def objective_function(**params):
        randname = rw.random_word() + str(random.randint(0, 100))
        sacredObj = ex.run(config_updates={'gpu': GPU,
                                           'epochs': 50,
                                           'manifold': manifold,
                                           'dataset': dataset,
                                           "h0": h0,
                                           'data_dir': data_dir,
                                           **params},
                           options={'--name': randname})
        return 1 - sacredObj.result
    res_gp = gp_minimize(objective_function, space, n_calls=100)
elif manifold == 'euclidean':
    space = [Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_euclidean')]
    @use_named_args(space)
    def objective_function(**params):
        randname = rw.random_word() + str(random.randint(0, 100))
        sacredObj = ex.run(config_updates={'gpu': GPU,
                                           'epochs': 50,
                                           'manifold': manifold,
                                           'dataset': dataset,
                                           "h0": h0,
                                           'data_dir': data_dir,
                                           **params},
                           options={'--name': randname})
        return 1 - sacredObj.result
    res_gp = gp_minimize(objective_function, space, n_calls=100)
elif manifold == 'oblique':
    space = [Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_euclidean'),
             Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_manifold'),
             Real(10 ** -8, 10 ** -3, "log-uniform", name='learning_rate_scale'),
             Real(1e-4, 1, "log-uniform",  name='omega')]
    
    @use_named_args(space)
    def objective_function(**params):
        randname = rw.random_word() + str(random.randint(0, 100))
        sacredObj = ex.run(config_updates={'gpu': GPU,
                                           'epochs': 50,
                                           'manifold': manifold,
                                           'dataset': dataset,
                                           "h0": h0,
                                           'data_dir': data_dir,
                                           **params},
                           options={'--name': randname})
        return 1 - sacredObj.result
    res_gp = gp_minimize(objective_function, space, n_calls=100)
elif manifold == 'oblique_penalized':
    space = [Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_euclidean'),
             Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_manifold'),
             Real(10 ** -8, 10 ** -3, "log-uniform", name='learning_rate_scale'),
             Real(1e-4, 1, "log-uniform", name='omega'),
             Real(1e-6, 1, "log-uniform", name='weight_decay')]


    @use_named_args(space)
    def objective_function(**params):
        randname = rw.random_word() + str(random.randint(0, 100))
        sacredObj = ex.run(config_updates={'gpu': GPU,
                                           'epochs': 50,
                                           'manifold': manifold,
                                           'dataset': dataset,
                                           "h0": h0,
                                           'data_dir': data_dir,
                                           **params},
                           options={'--name': randname})
        return 1 - sacredObj.result


    res_gp = gp_minimize(objective_function, space, n_calls=100)
elif manifold == 'stiefel_penalized':
    space = [Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_euclidean'),
             Real(10 ** -7, 10 ** -3, "log-uniform", name='learning_rate_manifold'),
             Real(10 ** -8, 10 ** -3, "log-uniform", name='learning_rate_scale'),
             Real(1e-6, 1, "log-uniform", name='weight_decay')]

    @use_named_args(space)
    def objective_function(**params):
        randname = rw.random_word() + str(random.randint(0, 100))
        sacredObj = ex.run(config_updates={'gpu': GPU,
                                           'epochs': 50,
                                           'manifold': manifold,
                                           'dataset': dataset,
                                           "h0": h0,
                                           'data_dir': data_dir,
                                           **params},
                           options={'--name': randname})
        return 1 - sacredObj.result


    res_gp = gp_minimize(objective_function, space, n_calls=100)
else:
    raise NotImplementedError

if gethostname != '***':
    server.stop()
api.update_status(status="Optimized hypers for {} net on {} with {}.".format(manifold, dataset, h0))
