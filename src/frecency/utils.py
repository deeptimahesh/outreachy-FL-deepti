# from utils import one_hot, normalize
# from model_checkpoint import ModelCheckpoint

import numpy as np

def one_hot(num_choices, vector):
    return np.eye(num_choices)[vector]

def flatten(X):
    X_flat = []

    for x in X:
        X_flat += list(x)

    return np.array(X_flat)

def normalize(X):
    X_flat = flatten(X)
    mu = X_flat.mean(axis=0)
    return [x - mu for x in X]

import numpy as np

class ModelCheckpoint:
    def __init__(self, metric_fn, data_generator, num_sampled=10000):
        self.best_model = None
        self.best_metric = -np.inf
        self.metric_fn = metric_fn
        self.num_sampled = num_sampled
        self.data_generator = data_generator
    
    def __call__(self, model):
        X_val, y_val = self.data_generator(self.num_sampled)
        metric = self.metric_fn(y_val, model.predict(X_val))
        
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_model = model
            print("[ModelCheckpoint] New best model with %.5f validation accuracy" % metric)
        else:
            print("validation: %.3f accuracy" % metric)