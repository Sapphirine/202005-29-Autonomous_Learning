# Bayesian Optimization Module for parameter searching. If the mode is set to tuning, this module will run Bayesian
# Optimization instead of training.


import numpy as np

from bayes_opt import BayesianOptimization
from main import train, validate

def best_parameters(T, K, alpha, lambda_u, rampup_length, weight_decay, ema_decay):
    print(T, K, alpha, lambda_u, rampup_length, weight_decay, ema_decay)
    args['T'] = T
    args['K'] = K
    args['alpha'] = alpha
    args['lambda_u'] = lambda_u
    args['rampup_length'] = rampup_length
    args['weight_decay'] = weight_decay
    train(datasetX, datasetU, model, ema_model, optimizer, 1, args)
    val_xe_loss, val_accuracy = validate(val_dataset, ema_model, 1, args, split='Validation')
    return val_accuracy


def random_search(T, K, alpha, lambda_u, rampup_length, weight_decay, ema_decay):
    global accbest
    param = {
        'T': np.float32(T),
        'K': int(np.around(K)),
        'alpha': np.float32(alpha),
        'lambda_u': np.float32(lambda_u),
        'rampup_length': int(np.around(rampup_length)),
        'weight_decay': np.float32(weight_decay),
        'ema_decay': np.float32(ema_decay),
    }
    print('\nSearch parameters %s' % param)
    accuracy = best_parameters(**param)
    if accuracy > accbest:
        accbest = accuracy
    return accuracy


def Bayesian_Optimization(params):
    global datasetX, datasetU, val_dataset, model, ema_model, optimizer, epoch, args
    datasetX, datasetU, val_dataset, model, ema_model, optimizer, epoch, args = params[0], params[1], params[2], \
                                                                                params[3], params[4], params[5], \
                                                                                params[6], params[7]
    global accbest
    accbest = 0.0
    NN_BAYESIAN = BayesianOptimization(
        random_search,
        {'T': (0, 1),
            'K': (1, 5),
            'alpha': (0.5, 1),
            'lambda_u': (50, 150),
            'rampup_length': (5, 25),
            'weight_decay': (0.01, 0.05),
            'ema_decay': (0.98, 1),
        }
    )
    NN_BAYESIAN.maximize(init_points=5, n_iter=10, acq='ei', xi=0.0)