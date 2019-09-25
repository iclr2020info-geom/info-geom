import sys, os
from socket import gethostname
from pymongo import MongoClient, DESCENDING
from experiment import ex
import random
from random_words import RandomWords
import tweepy

dataset = sys.argv[1]
manifold = sys.argv[2]
h0 = float(sys.argv[3])

if gethostname() != '***':
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
    client = MongoClient('127.0.0.1', server.local_bind_port)
    GPU = 0
    if gethostname() == '***':
        project_dir = '/***/home/***/projects'
    else:
        project_dir = '/***/home/***/projects'
    data_dir = os.path.join(project_dir, 'data')
else:
    from ops.utils import bestGPU
    project_dir = '/home/***/projects/isonetry/'
    data_dir = os.path.join(project_dir,'data')
    client = MongoClient('127.0.0.1', 27017)
    GPU = bestGPU()

MONGO_DB = "isonetry-hyperparams2"
runs = client[MONGO_DB]['runs']

hypers = runs.find({
                    "$and": [{"config.dataset": dataset}, {"config.h0": h0}, {"config.manifold": manifold}]},
                        {
                            "config.learning_rate_euclidean": 1, "config.learning_rate_factor": 1,
                            "config.learning_rate_manifold": 1, "config.learning_rate_scale": 1, "config.omega": 1
                        }
                ).sort("result", DESCENDING).limit(1)[0]['config']

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
    ex.run(named_configs=[manifold+str(2)], config_updates={'gpu': GPU,
                                                      'dataset':dataset,
                                                      'h0': h0,
                                                      'learning_rate_euclidean': hypers['learning_rate_euclidean'],
                                                      'learning_rate_manifold': hypers['learning_rate_manifold'],
                                                      'learning_rate_scale': hypers['learning_rate_manifold'],
                                                            'data_dir': data_dir
                                                        }, options={'--name': randname})
elif manifold == 'euclidean':
    ex.run( config_updates={'gpu': GPU,
                                                     'dataset': dataset,
                                                     'h0': h0,
                                                     'learning_rate_euclidean': hypers['learning_rate_euclidean'],
                                                     }, options={'--name': randname})

elif manifold == 'oblique':
    ex.run(named_configs=[manifold], config_updates={'gpu': GPU,
                                                      'dataset':dataset,
                                                      'h0': h0,
                                                      'learning_rate_euclidean': hypers['learning_rate_euclidean'],
                                                      'learning_rate_manifold': hypers['learning_rate_manifold'],
                                                      'learning_rate_scale': hypers['learning_rate_manifold'],
                                                      'omega': hypers['omega'],
                                                     'data_dir': data_dir
                                                        }, options={'--name': randname})
elif manifold == 'onlique_penalized':
    ex.run(named_configs=[manifold], config_updates={'gpu': GPU,
                                                     'dataset': dataset,
                                                     'h0': h0,
                                                     'learning_rate_euclidean': hypers['learning_rate_euclidean'],
                                                     'learning_rate_manifold': hypers['learning_rate_manifold'],
                                                     'learning_rate_scale': hypers['learning_rate_manifold'],
                                                     'omega': hypers['omega'],
                                                     'weight_decay': hypers['weight_decay']
                                                     }, options={'--name': randname})
elif manifold == 'stiefel_penalized':
    ex.run(named_configs=[manifold], config_updates={'gpu': GPU,
                                                              'dataset': dataset,
                                                              'h0': h0,
                                                              'learning_rate_euclidean': hypers[
                                                                  'learning_rate_euclidean'],
                                                              'learning_rate_manifold': hypers[
                                                                  'learning_rate_manifold'],
                                                              'learning_rate_scale': hypers['learning_rate_manifold'],
                                                              'weight_decay': hypers['weight_decay'],
                                                                'data_dir': data_dir
                                                              }, options={'--name': randname})
else:
    raise NotImplementedError

if gethostname!='***':
    server.stop()
api.update_status(status="Finished one {} net on {} with {}.".format(manifold, dataset, h0))
