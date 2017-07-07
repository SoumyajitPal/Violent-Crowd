from sklearn.model_selection import KFold
import numpy as np

def k_fold(k, data, labels):
    kfold = KFold(n_splits=5)
    indices = []

    for train, test in kfold.split(data):
        indices.append((train, test))

    return indices    
