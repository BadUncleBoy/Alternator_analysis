import numpy as np
def evaluate_precision(predicate_class, lable_class):
    predicate_class = np.argmax(predicate_class, axis=1)
    lable_class     = np.argmax(lable_class, axis=1)
    precision       = np.sum(predicate_class == lable_class)/predicate_class.shape[0]
    return precision

def evaluate_error_rate(predicate_regree, lable_class, lable_regree):
    predicate_regree = lable_class * predicate_regree
    predicate_regree = np.max(predicate_regree, axis=1)
    lable_regree     = np.max(lable_regree, axis=1)
    error_rate       = np.abs((predicate_regree - lable_regree)/lable_regree)
    return 1 - np.mean(error_rate)
