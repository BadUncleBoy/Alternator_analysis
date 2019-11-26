import numpy as np
def evaluate_precision(predicate, class_lable):
    predicate_lable = np.argmax(predicate, axis=1)
    true_lable      = np.argmax(class_lable, axis=1)
    precision       = np.sum(predicate_lable == true_lable)/predicate.shape[0]
    return precision

def evaluate_error_rate(predicate, true_lable, regree_lable):
    predicate     = true_lable * predicate
    predicate     = np.max(predicate, axis=1)
    regree_lable  = np.max(regree_lable, axis=1)
    error_rate    = np.abs((predicate - regree_lable)/regree_lable)
    return 1 - np.mean(error_rate)
